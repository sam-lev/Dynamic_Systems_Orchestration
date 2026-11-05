#!/usr/bin/env python3
# test_rave_layer.py -- Audible Lava / DSO v4.11
# Tests for rave_layer.py and the RAVE tab in controller.py. No SuperCollider,
# camera or pygame: OSC sends go to a recorder and the keyboard controller is
# driven through a stand-in pygame module. Runs standalone:
#   python test_rave_layer.py

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import types

from rave_layer import (MODELS, MODEL_BY_KEY, RaveLayer, install,
                        parse_model_filename, model_pan, per_model_amp, dry_gain,
                        RAVE_BUS, SPLIT_NODE, MODEL_NODE_BASE)
from controller import (ControllerEvent, RAVE_TOGGLE_MODEL, RAVE_TOGGLE, RAVE_SET,
                        RAVE_ALL_OFF, RAVE_RELOAD, RAVE_MODEL_KEYS, MODE_TOGGLE,
                        SELECT_SLOT, NOTE_ON, PERFORMANCE_MODE, EDIT_MODE, RAVE_MODE)

HERE = os.path.dirname(os.path.abspath(__file__))

# The twelve files Sam listed, verbatim.
SHIPPED_FILES = """
birds_dawnchorus_b2048_r48000_z8.ts
crozzoli_bigensemblesmusic_18d.ts
guitar_iil_b2048_r48000_z16.ts
humpbacks_pondbrain_b2048_r48000_z20.ts
magnets_b2048_r48000_z8.ts
marinemammals_pondbrain_b2048_r48000_z20.ts
mrp_strengjavera_b2048_r44100_z16.ts
organ_archive_b2048_r48000_z16.ts
voice-multi-b2048-r48000-z11.ts
voice_vctk_b2048_r44100_z22.ts
voice_vocalset_b2048_r48000_z16.ts
water_pondbrain_b2048_r48000_z16.ts
""".split()


# ── fixtures ─────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.msgs = []

    def __call__(self, addr, args):
        self.msgs.append((addr, list(args)))

    def send_message(self, addr, args):
        self(addr, args)

    def of(self, addr):
        return [a for ad, a in self.msgs if ad == addr]

    def clear(self):
        self.msgs = []


class Voices:
    def __init__(self, osc):
        self.osc = osc


def kv(pairs):
    """Flat [name, value, ...] -> {name: value}."""
    p = list(pairs)
    return {p[i]: p[i + 1] for i in range(0, len(p) - 1, 2)}


def snew(args):
    """/s_new args [synthdef, node, addAction, target, k, v, ...] -> {k: v}."""
    return kv(args[4:])


def nset(args):
    """/n_set args [node, k, v, ...] -> {k: v}."""
    return kv(args[1:])


def write_status(path, loaded=(), missing=(), failed=(), server_rate=48000, available=True):
    models = {}
    for m in MODELS:
        if m.key in loaded:
            models[m.key] = {"status": "loaded", "rate": m.rate or 48000, "minBufferSize": 2048, "error": ""}
        elif m.key in missing:
            models[m.key] = {"status": "missing", "rate": None, "minBufferSize": None, "error": "file not found"}
        elif m.key in failed:
            models[m.key] = {"status": "failed", "rate": None, "minBufferSize": None, "error": "boom"}
        else:
            models[m.key] = {"status": "unknown", "rate": None, "minBufferSize": None, "error": ""}
    with open(path, "w") as f:
        json.dump({"nn_available": available, "sampleRate": server_rate, "bufferSize": -1,
                   "written": "test", "models": models}, f)


def make_layer(rec, status=None, **kw):
    """A layer wired to a recorder, with an optional status file."""
    path = status or os.path.join(tempfile.mkdtemp(), "rave_status.json")
    t = [1000.0]
    layer = RaveLayer(send=rec, status_path=path, clock=lambda: t[0], **kw)
    layer._t = t
    return layer


# ── stand-in pygame for the controller tests ──────────────────────────────────

class _Ev:
    def __init__(self, type, key=None):
        self.type = type; self.key = key


def fake_pygame():
    pg = types.ModuleType("pygame")
    pg.QUIT, pg.KEYDOWN, pg.KEYUP = 256, 768, 769
    pg.K_TAB, pg.K_SPACE, pg.K_EQUALS, pg.K_MINUS = 9, 32, 61, 45
    pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET = 91, 93
    pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT = 1073741906, 1073741905, 1073741904, 1073741903
    pg.RESIZABLE = 16
    pg.error = type("error", (Exception,), {})
    pg.init = lambda: None
    pg.quit = lambda: None
    pg.event = types.SimpleNamespace(queue=[], get=lambda: pg.event.pop_all())
    pg.event.pop_all = lambda: [pg.event.queue.pop(0) for _ in range(len(pg.event.queue))]
    pg.display = types.SimpleNamespace(set_mode=lambda *a, **k: (_ for _ in ()).throw(pg.error()),
                                       set_caption=lambda *a: None, flip=lambda: None)
    pg.font = types.SimpleNamespace(SysFont=lambda *a, **k: None)
    return pg


def make_keyboard():
    pg = fake_pygame()
    sys.modules["pygame"] = pg
    from controller import KeyboardController
    kb = KeyboardController(init_display=False)
    kb._pg_fake = pg
    return kb, pg


def press(pg, ch=None, key=None, up=False):
    pg.event.queue.append(_Ev(pg.KEYUP if up else pg.KEYDOWN, key if key is not None else ord(ch)))


# ── tests: model table ───────────────────────────────────────────────────────

def test_model_table_matches_shipped_files():
    assert len(MODELS) == 12
    assert [m.file for m in MODELS] == SHIPPED_FILES, "file list must be exactly the shipped twelve"
    assert len({m.key for m in MODELS}) == 12 and all(re.fullmatch(r"[a-z]+", m.key) for m in MODELS)
    assert len(RAVE_MODEL_KEYS) == 12 and len(set(RAVE_MODEL_KEYS)) == 12
    assert MODEL_BY_KEY["birds"].synthdef == "raveXfer_birds"
    assert MODEL_BY_KEY["birds"].path.endswith(os.path.join("RAVE_PreTrained_Models", SHIPPED_FILES[0]))


def test_parse_model_filename():
    assert parse_model_filename("birds_dawnchorus_b2048_r48000_z8.ts") == (2048, 48000, 8)
    assert parse_model_filename("voice-multi-b2048-r48000-z11.ts") == (2048, 48000, 11)
    assert parse_model_filename("crozzoli_bigensemblesmusic_18d.ts") == (None, None, None)
    assert MODEL_BY_KEY["mrp"].rate == 44100 and MODEL_BY_KEY["vctk"].rate == 44100
    assert MODEL_BY_KEY["humpbacks"].latent == 20
    assert all(m.buffer in (2048, None) for m in MODELS)


def test_scd_mirrors_python_model_list():
    """blob_synth_rave.scd must list the same key -> file pairs, in order."""
    path = os.path.join(HERE, "blob_synth_rave.scd")
    if not os.path.isfile(path):
        return                                       # skipped where the .scd is not alongside
    src = open(path).read()
    pairs = re.findall(r"\[\\(\w+),\s*\"([^\"]+\.ts)\"\]", src)
    assert pairs == [(m.key, m.file) for m in MODELS], pairs
    assert "~raveBus        = ~raveBus        ? %d" % RAVE_BUS in src
    assert "SynthDef(\\raveSplit" in src and "raveXfer_" in src
    assert "\\NN.asClass" in src, "nn.ar must be reached through asClass so the file compiles without it"


def test_level_and_placement_helpers():
    assert dry_gain(False, 3, 0.7) == 1.0 and dry_gain(True, 0, 0.7) == 1.0
    assert abs(dry_gain(True, 1, 0.7) - 0.3) < 1e-9
    assert per_model_amp(0.7, 0) == 0.0 and abs(per_model_amp(0.7, 1) - 0.7) < 1e-9
    assert abs(per_model_amp(0.7, 4) - 0.35) < 1e-9
    assert model_pan(0) == -0.6 and model_pan(11) == 0.6 and abs(model_pan(5) + model_pan(6)) < 1e-9
    assert model_pan(0, 1) == 0.0


# ── tests: the layer ─────────────────────────────────────────────────────────

def test_install_spawns_split_before_master_as_passthrough():
    rec = Recorder()
    layer = install(Voices(rec), master_node=900, send=rec,
                    status_path=os.path.join(tempfile.mkdtemp(), "rave_status.json"))
    sn = rec.of("/s_new")
    assert len(sn) == 1 and sn[0][:4] == ["raveSplit", SPLIT_NODE, 2, 900], "addBefore the master"
    d = snew(sn[0]); assert d["raveBus"] == RAVE_BUS and d["dry"] == 1.0
    assert layer.split_spawned and not layer.enabled and layer.n_active == 0
    assert nset(rec.of("/n_set")[-1])["dry"] == 1.0
    # no master -> tail of group 0
    rec2 = Recorder()
    install(Voices(rec2), master_node=None, send=rec2,
            status_path=os.path.join(tempfile.mkdtemp(), "rave_status.json"))
    assert rec2.of("/s_new")[0][:4] == ["raveSplit", SPLIT_NODE, 1, 0]


def test_models_only_sound_when_layer_is_on():
    rec = Recorder(); layer = make_layer(rec, master_node=900, mix=0.7)
    layer.spawn_split(); rec.clear()
    layer.toggle_model(0)                              # birds selected, layer OFF
    assert layer.active[0] and rec.of("/s_new") == [], "no node while the layer is off"
    assert nset(rec.of("/n_set")[-1])["dry"] == 1.0
    rec.clear()
    layer.set_enabled(True)
    sn = rec.of("/s_new"); assert len(sn) == 1
    assert sn[0][:4] == ["raveXfer_birds", MODEL_NODE_BASE + 0, 2, 900]
    d = snew(sn[0])
    assert d["inBus"] == RAVE_BUS and d["out"] == 0 and d["gate"] == 1
    assert abs(d["amp"] - 0.7) < 1e-9 and d["pan"] == model_pan(0) and d["inGain"] == 1.0
    dry = [nset(a) for a in rec.of("/n_set") if a[0] == SPLIT_NODE][-1]["dry"]
    assert abs(dry - 0.3) < 1e-9
    assert "RAVE on" in layer.hud_line() and "birds" in layer.hud_line()
    rec.clear()
    layer.set_enabled(False)
    assert rec.of("/n_set")[0] == [MODEL_NODE_BASE, "gate", 0], "layer off gates the node (fade, self-free)"
    assert layer.active[0], "selection survives turning the layer off"
    assert layer.spawned_nodes == {}
    assert layer.hud_line() == "RAVE off"


def test_bank_levels_follow_active_count():
    rec = Recorder(); layer = make_layer(rec, master_node=None, mix=0.8, enabled=True)
    layer.spawn_split()
    layer.toggle_model(7)                              # organ
    layer.toggle_model(11)                             # water
    rec.clear()
    layer._push_amps()
    amps = {a[0]: nset(a)["amp"] for a in rec.of("/n_set")}
    assert set(amps) == {MODEL_NODE_BASE + 7, MODEL_NODE_BASE + 11}
    assert all(abs(v - 0.8 / math.sqrt(2)) < 1e-3 for v in amps.values())
    # placement without a master: tail of group 0
    rec.clear(); layer.toggle_model(2)
    assert rec.of("/s_new")[0][:4] == ["raveXfer_guitar", MODEL_NODE_BASE + 2, 1, 0]
    # toggling one off gates only that node and re-levels the rest
    rec.clear(); layer.toggle_model(7)
    assert rec.of("/n_set")[0] == [MODEL_NODE_BASE + 7, "gate", 0]
    rest = {a[0] for a in rec.of("/n_set") if "amp" in nset(a)}
    assert rest == {MODEL_NODE_BASE + 11, MODEL_NODE_BASE + 2}
    layer.all_off()
    assert layer.n_active == 0 and layer.spawned_nodes == {}
    assert nset(rec.of("/n_set")[-1])["dry"] == 1.0


def test_mix_and_in_gain():
    rec = Recorder(); layer = make_layer(rec, master_node=900, enabled=True, mix=0.5)
    layer.spawn_split(); layer.toggle_model(3); rec.clear()
    layer.set_mix(1.0)
    sets = {a[0]: nset(a) for a in rec.of("/n_set")}
    assert abs(sets[MODEL_NODE_BASE + 3]["amp"] - 1.0) < 1e-9 and sets[SPLIT_NODE]["dry"] == 0.0
    assert layer.set_mix(1.7) == 1.0 and layer.set_mix(-1) == 0.0
    rec.clear()
    assert abs(layer.set_in_gain(2.0) - 2.0) < 1e-9
    assert nset(rec.of("/n_set")[0])["inGain"] == 2.0
    assert layer.set_in_gain(99) == 4.0 and layer.set_in_gain(0) == 0.1


def test_controller_events_drive_the_layer():
    rec = Recorder(); layer = make_layer(rec, master_node=900, mix=0.7)
    layer.spawn_split()
    assert layer.handle_controller_event(ControllerEvent(RAVE_TOGGLE_MODEL, extras={"index": 4}))
    assert layer.active[4] and layer.spawned_nodes == {}
    assert layer.handle_controller_event(ControllerEvent(RAVE_TOGGLE))
    assert layer.enabled and 4 in layer.spawned_nodes
    assert layer.handle_controller_event(ControllerEvent(RAVE_SET, param="mix", value=+0.05,
                                                         extras={"relative": True}))
    assert abs(layer.mix - 0.75) < 1e-9
    assert layer.handle_controller_event(ControllerEvent(RAVE_SET, param="mix", value=0.2,
                                                         extras={"relative": False}))
    assert abs(layer.mix - 0.2) < 1e-9
    g0 = layer.in_gain
    assert layer.handle_controller_event(ControllerEvent(RAVE_SET, param="in_gain", value=+1.0))
    assert abs(layer.in_gain - 2 * g0) < 1e-9
    assert layer.handle_controller_event(ControllerEvent(RAVE_ALL_OFF)) and layer.n_active == 0
    assert layer.handle_controller_event(ControllerEvent(RAVE_RELOAD))
    assert not layer.handle_controller_event(ControllerEvent(NOTE_ON, slot=0, note=60))
    assert not layer.handle_controller_event(ControllerEvent(RAVE_TOGGLE_MODEL, extras={"index": 99})) is None
    assert not layer.toggle_model(99) and not layer.toggle_model(-1)


def test_status_file_and_panel():
    d = tempfile.mkdtemp(); path = os.path.join(d, "rave_status.json")
    rec = Recorder()
    layer = make_layer(rec, status=path, master_node=900)
    assert not layer.status.models
    rows = layer.panel_lines()
    assert rows[0][1] == "title" and "OFF" in rows[0][0]
    assert any("rave_status.json not found" in t for t, k in rows if k == "warn")
    model_rows = [r for r in rows if r[1] in ("on", "off", "unverified", "missing", "failed")]
    assert len(model_rows) == 12 and all(r[1] == "unverified" for r in model_rows)
    assert model_rows[0][0].startswith(" a  [ ]  birds")
    # SC writes the file: loaded / missing / failed and a 44.1 k mismatch
    write_status(path, loaded=["birds", "organ", "mrp"], missing=["water"], failed=["vctk"])
    layer._t[0] += 5.0; layer.update()
    assert layer.status.available and layer.status.server_rate == 48000
    rows = {r[0].split()[0] + "|" + r[0][2:].split("[")[1][0:1]: r for r in layer.panel_lines() if r[1] in
            ("on", "off", "unverified", "missing", "failed")}
    kinds = {k: layer.model_line(i, RAVE_MODEL_KEYS[i])[1] for i, k in enumerate(m.key for m in MODELS)}
    assert kinds["birds"] == "off" and kinds["water"] == "missing" and kinds["vctk"] == "failed"
    assert kinds["guitar"] == "unverified"
    mrp_line = layer.model_line(6, "j")[0]
    assert "44k!" in mrp_line, mrp_line                      # trained at 44.1 k, server at 48 k
    assert "48k " in layer.model_line(0, "a")[0] or "48k" in layer.model_line(0, "a")[0]
    layer.toggle_model(0)
    assert layer.model_line(0, "a")[1] == "on" and "[x]" in layer.model_line(0, "a")[0]
    assert layer.last_msg.endswith("on")
    layer.toggle_model(11)
    assert "missing in SC" in layer.last_msg
    # a rewrite is picked up by the poll, an unchanged file is not re-read
    write_status(path, loaded=[m.key for m in MODELS])
    layer._t[0] += 5.0; layer.update()
    assert all(s.status == "loaded" for s in layer.status.models.values())
    layer.reload_status(); assert "12 model(s) loaded" in layer.last_msg
    # nn.ar absent is called out
    write_status(path, available=False)
    layer._t[0] += 5.0; layer.update()
    assert any("nn.ar not installed" in t for t, k in layer.panel_lines() if k == "warn")


def test_close_then_free_split_ordering():
    rec = Recorder(); layer = make_layer(rec, master_node=900, enabled=True)
    layer.spawn_split(); layer.toggle_model(1); layer.toggle_model(2); rec.clear()
    layer.close()
    gates = [a for a in rec.of("/n_set") if a[1:] == ["gate", 0]]
    assert {a[0] for a in gates} == {MODEL_NODE_BASE + 1, MODEL_NODE_BASE + 2}
    assert nset(rec.of("/n_set")[-1])["dry"] == 1.0 and layer.split_spawned
    assert rec.of("/n_free") == []
    layer.free_split()
    assert rec.of("/n_free") == [[SPLIT_NODE]] and not layer.split_spawned


# ── tests: the RAVE tab in the keyboard controller ───────────────────────────

def test_tab_cycle_only_includes_rave_when_enabled():
    kb, pg = make_keyboard()
    press(pg, key=pg.K_TAB); kb.poll(); assert kb.mode == EDIT_MODE
    press(pg, key=pg.K_TAB); kb.poll(); assert kb.mode == PERFORMANCE_MODE
    kb.enable_rave_tab(lambda: [("RAVE", "title")])
    press(pg, key=pg.K_TAB); kb.poll(); assert kb.mode == EDIT_MODE
    press(pg, key=pg.K_TAB); evs = kb.poll(); assert kb.mode == RAVE_MODE
    assert evs[-1].type == MODE_TOGGLE and kb.last_event_msg == "mode -> RAVE"
    press(pg, key=pg.K_TAB); kb.poll(); assert kb.mode == PERFORMANCE_MODE


def test_rave_tab_keys():
    kb, pg = make_keyboard()
    kb.enable_rave_tab(lambda: [])
    kb.mode = RAVE_MODE
    for i, ch in enumerate(RAVE_MODEL_KEYS):
        press(pg, ch)
    evs = kb.poll()
    assert [e.type for e in evs] == [RAVE_TOGGLE_MODEL] * 12
    assert [e.extras["index"] for e in evs] == list(range(12))
    press(pg, key=pg.K_SPACE); press(pg, "z"); press(pg, "r")
    evs = kb.poll()
    assert [e.type for e in evs] == [RAVE_TOGGLE, RAVE_ALL_OFF, RAVE_RELOAD]
    press(pg, key=pg.K_UP); press(pg, key=pg.K_DOWN); press(pg, key=pg.K_EQUALS)
    press(pg, key=pg.K_MINUS); press(pg, key=pg.K_RIGHT); press(pg, key=pg.K_LEFT)
    evs = kb.poll()
    assert all(e.type == RAVE_SET for e in evs)
    assert [(e.param, e.value) for e in evs] == [("mix", 0.05), ("mix", -0.05), ("mix", 0.25),
                                                  ("mix", -0.25), ("in_gain", 0.5), ("in_gain", -0.5)]
    # number keys still select slots in the RAVE tab; key-ups and other keys do nothing
    press(pg, "3"); press(pg, "a", up=True); press(pg, "q")
    evs = kb.poll()
    assert [e.type for e in evs] == [SELECT_SLOT] and kb.selected_slot == 2
    # and in PERFORMANCE the home row is still a piano
    kb.mode = PERFORMANCE_MODE
    press(pg, "s")
    assert [e.type for e in kb.poll()] == [NOTE_ON]


def test_layer_panel_renders_without_a_display():
    """The panel callable installed by the main loop is what the tab draws;
    it must not raise for any state."""
    rec = Recorder(); layer = make_layer(rec, master_node=900)
    kb, pg = make_keyboard()
    kb.enable_rave_tab(layer.panel_lines, RAVE_MODEL_KEYS)
    kb.mode = RAVE_MODE
    press(pg, "a"); press(pg, key=pg.K_SPACE)
    for ev in kb.poll():
        layer.handle_controller_event(ev)
    assert layer.enabled and layer.active[0]
    rows = kb.rave_panel()
    assert any(k == "on" for _t, k in rows) and rows[0][0].startswith("RAVE timbre transfer   ON")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    passed = failed = 0
    for t in TESTS:
        try:
            t(); passed += 1; print(f"  ok   {t.__name__}")
        except Exception as exc:
            failed += 1; print(f"  FAIL {t.__name__}: {exc!r}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
