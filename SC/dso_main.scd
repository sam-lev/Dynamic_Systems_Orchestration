// dso_main.scd -- Audible Lava / DSO: one-shot SuperCollider bring-up
//
// Evaluate this ONE block (cursor anywhere inside the outer parentheses,
// Cmd-Return). It loads the SuperCollider half of the pipeline in order and
// waits for each file to be truly ready before starting the next:
//
//   1  blob_synth.scd        boots the server (48 kHz, OMP/KMP env set first),
//                            fills the 16 wavetables, classic SynthDefs
//   2  blob_synth_hits.scd   \mpeHit, \preSound
//   3  melody_hit.scd        \melodyHit
//   4  blob_synth_rich.scd   rich palette + \lavaMaster
//   5  blob_synth_rave.scd   RAVE model descriptions, \raveXfer_<key>, \raveSplit,
//                            rave_status.json   (the slow step: ~2-5 s per model)
//   6  rave_bench.scd        optional (set ~dsoBench = true)
//
// WHY IT POLLS: every file does its real work asynchronously inside
// s.waitForBoot (buffer fills, s.sync, model loads), so `.load` returns long
// before the file is done. Each step therefore waits until that file's LAST
// SynthDef has been added (its "done" marker), then s.sync's so the server
// has it too. The markers are cleared first, so re-running in the same
// session really waits again. No other file needs any change.
//
// Put this file in the same folder as the other .scd files (your SC/ folder).
// Then start Python as usual -- or set ~dsoPython to launch it from here.
//
// Options (set before evaluating the block, all optional):
//   ~dsoFreshBoot = true;     // quit a running server first (default true:
//                             // nn.ar model ids live on one server, and Python
//                             // node ids must match a fresh /g_freeAll)
//   ~dsoBench     = false;    // also load rave_bench.scd
//   ~dsoRave      = true;     // false = skip the RAVE file (no nn.ar needed)
//   ~dsoDir       = "...";    // folder of the .scd files if not this file's
//   ~dsoPython    = "/Users/multivax/.conda/envs/dso/bin/python";  // launch
//   ~dsoPyDir     = "...";    // where audible_lava.py lives (default: parent of SC/)
//   ~dsoPyArgs    = "";       // e.g. "--no-melody"
//
// Launching Python from here: its output lands in the SC post window, and
// macOS asks for camera access on behalf of SuperCollider (the parent app) the
// first time. Starting Python in your terminal is the safer default.

(
var dir, steps, waitFor, t0;

dir = ~dsoDir ?? {
    thisProcess.nowExecutingPath !? { |p| p.dirname }
} ?? {
    "dso_main.scd: save this file next to the other .scd files (or set ~dsoDir)".error;
    nil
};

~dsoFreshBoot = ~dsoFreshBoot ? true;
~dsoBench     = ~dsoBench     ? false;
~dsoRave      = ~dsoRave      ? true;

// [file, done-marker SynthDef (the last one the file adds), timeout s]
steps = [
    ["blob_synth.scd",      \bassStutter, 60],
    ["blob_synth_hits.scd", \preSound,    30],
    ["melody_hit.scd",      \melodyHit,   30],
    ["blob_synth_rich.scd", \lavaMaster,  60]
];
if (~dsoRave)  { steps = steps.add(["blob_synth_rave.scd", \raveSplit,  300]) };
if (~dsoBench) { steps = steps.add(["rave_bench.scd",      \benchMeter, 120]) };

waitFor = { |name, timeout|
    var t = 0;
    while { SynthDescLib.global.at(name).isNil and: { t < timeout } } {
        0.25.wait; t = t + 0.25;
    };
    SynthDescLib.global.at(name).notNil
};

if (dir.notNil) {
    Routine({
        var ok = true;
        t0 = Main.elapsedTime;
        "\n==== dso_main.scd: bringing up Audible Lava / DSO ====".postln;
        ("     folder: " ++ dir).postln;

        // 0. fresh server (blob_synth.scd sets the OMP/KMP env and boots it)
        if (~dsoFreshBoot and: { s.serverRunning }) {
            "     quitting the running server for a clean start ...".postln;
            s.quit;
            while { s.serverRunning } { 0.1.wait };
            1.0.wait;
        };

        steps.do { |st|
            var file = st[0], marker = st[1], timeout = st[2], path, t1;
            if (ok) {
                path = dir +/+ file;
                if (File.exists(path).not) {
                    ("  !! " ++ file ++ " not found in " ++ dir).error;
                    ok = false;
                } {
                    t1 = Main.elapsedTime;
                    ("  -> " ++ file).postln;
                    SynthDescLib.global.synthDescs.removeAt(marker);   // re-runs really wait
                    path.load;
                    // the server may still be booting after step 1's .load
                    while { s.serverRunning.not and: { (Main.elapsedTime - t1) < 30 } } { 0.2.wait };
                    if (waitFor.(marker, timeout)) {
                        s.sync;
                        ("     ok  % (%s)".format(file, (Main.elapsedTime - t1).round(0.1))).postln;
                    } {
                        ("  !! % did not finish within % s (no \\% SynthDef) -- see the errors above"
                         .format(file, timeout, marker)).error;
                        ok = false;
                    };
                };
            };
        };

        if (ok) {
            // sanity checks that would otherwise only show up as silence
            if (~wtBuffers.isNil) { "  !! wavetables missing: VOsc voices will be silent".error };
            if (~dsoRave) {
                ("     RAVE: % of % models loaded, buffer %"
                 .format((~raveLoaded ? []).size, (~raveModels ? []).size, ~raveBufferSize)).postln;
            };
            ("==== dso_main.scd: ready in %s ====".format((Main.elapsedTime - t0).round(0.1))).postln;

            if (~dsoPython.notNil) {
                var pyDir = ~dsoPyDir ?? { dir.dirname };
                var cmd = "cd % && % audible_lava.py %"
                          .format(pyDir.shellQuote, ~dsoPython.shellQuote, ~dsoPyArgs ? "");
                ("     launching: " ++ cmd).postln;
                cmd.unixCmd({ |code| ("audible_lava.py exited (" ++ code ++ ")").postln });
            } {
                "     now start Python:  python audible_lava.py".postln;
            };
        } {
            "==== dso_main.scd: stopped -- fix the step above and evaluate again ====".postln;
        };
    }).play(AppClock);
};
)

/*
How to use it:

Put it in your SC/ folder next to the other .scd files and save it. It finds the other files from its own location, so it must be saved, not an untitled document. If it can’t find them, set ~dsoDir = "path/to/SC"; first.
Put the cursor inside the outer parentheses and press Cmd-Return.
When it posts ==== dso_main.scd: ready ====, run python audible_lava.py in your terminal.

What it does:

Fresh server first. If a server is already running, it quits it and lets blob_synth.scd boot a new one, so the OMP/KMP settings take effect and nn.ar and Python start from a clean state. Set ~dsoFreshBoot = false to keep the running server.
Loads the files in your order: blob_synth, blob_synth_hits, melody_hit, blob_synth_rich, blob_synth_rave. After each, it waits until that file’s last SynthDef exists, then syncs with the server. Each step posts ok <file> (Ns).
Stops on failure. If a file doesn’t finish in time (60 s, or 300 s for the RAVE file), it reports which one and stops instead of loading the next file onto a broken setup. Fix it and evaluate again.
Checks at the end that the wavetables exist, and reports how many RAVE models loaded and at what buffer size.

Options, set before evaluating:

~dsoBench = true; also loads rave_bench.scd.
~dsoRave = false; skips RAVE entirely.
~dsoPython = "/Users/multivax/.conda/envs/dso/bin/python"; launches audible_lava.py from SuperCollider at the end, with ~dsoPyArgs for flags. I’d keep starting Python from the terminal, though. Launched this way, its output lands in the SC post window, and macOS will ask for camera permission on SuperCollider’s behalf.
*/