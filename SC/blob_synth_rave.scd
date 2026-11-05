// blob_synth_rave.scd -- Audible Lava / DSO v4.11 "RAVE timbre transfer"
//
// ADDITIVE. Evaluate AFTER blob_synth_rich.scd (the server is booted by then).
// Nothing in the other .scd files is touched; with this file never evaluated
// the system runs exactly as v4.10 (Python's rave_layer sends /s_new for
// SynthDefs that then do not exist, which only prints a line here).
//
// Load order:  blob_synth.scd -> blob_synth_hits.scd -> melody_hit.scd
//              -> blob_synth_rich.scd -> blob_synth_rave.scd
//
// WHAT THIS FILE DOES
//
//   1. Loads the description of every pre-trained RAVE model it finds in
//      ~raveModelDir with NN.load (nn.ar, https://github.com/elgiano/nn.ar).
//      RAVE = Realtime Audio Variational autoEncoder (Caillon & Esling, IRCAM).
//      Loading a description means scsynth opens the torchscript once, so this
//      takes a few seconds per model on first evaluation; progress is posted.
//   2. Builds one \raveXfer_<key> SynthDef per model that loaded. nn.ar
//      resolves the model by index when the SynthDef is BUILT, which is why
//      there is one SynthDef per model rather than one with a selector --
//      and why each node holds its own independent instance of its model
//      (twelve models on at once would be twelve inference threads).
//   3. Builds \raveSplit, the node that copies bus 0 to a private bus for the
//      models and scales the dry mix.
//   4. Writes rave_status.json next to the models so Python's GUI can show
//      which models loaded, their trained sample rate, and the server's rate.
//
//   Python (rave_layer.py) spawns the nodes: the split once at startup (node
//   910, added BEFORE \lavaMaster 900 so the space hears the transferred
//   timbre), then one model node (920 + index) per model the player toggles
//   on from the keyboard window's RAVE tab. Node argument contract:
//
//     \raveSplit          raveBus dry
//     \raveXfer_<key>     inBus out amp inGain pan gate fadeIn fadeOut
//
// REQUIREMENTS
//   * nn.ar in the Extensions folder (release build for this macOS/arch).
//     First run on macOS:  NN.deQuarantine; s.reboot;   (unsigned binaries)
//   * The .ts files in ~raveModelDir (set ~raveModelDir before evaluating to
//     point elsewhere; the Python side reads AUDIBLE_LAVA_RAVE_DIR).
//   * s.options.blockSize <= the model's buffer (2048): default 64 is fine.
//   * CPU: one RAVE v2 model at 2048/48k is roughly 15-30 ms of inference per
//     43 ms buffer on a 2016 i5 (nn.ar README); comfortable for 2-4 models on
//     Apple silicon, marginal on a Raspberry Pi 5 (which also needs nn.ar
//     built from source against an arm64 libtorch).
//
// NOTES
//   * nn.ar does not resample: a model trained at 44.1 kHz run at 48 kHz is
//     ~8.8 % off-rate (slight spectral tilt). Two of the twelve are 44.1 k;
//     the post window and the Python GUI flag them.
//   * Each model node runs inference on its own thread (bufferSize > 0). It
//     is silent for a moment after /s_new while the thread loads its model;
//     the fade-in covers that. Destroying the node waits for that thread,
//     which can block the DSP chain briefly -- so Python gates nodes off
//     (fade-out, then self-free) rather than freeing them mid-sound.
//   * The models are mono. The split feeds them the L+R sum; each model's
//     output is panned to a fixed position from Python (bank spread).
//
// Every reference to the extension goes through \NN.asClass so this file
// compiles when nn.ar is absent (a literal missing class name is a whole-file
// compile error).

~raveModelDir   = ~raveModelDir   ? "~/Documents/RAVE_PreTrained_Models".standardizePath;
~raveBufferSize = ~raveBufferSize ? -1;     // -1 = each model's own minimum (2048 here);
                                            // 4096 trades latency for smoother CPU use
~raveBus        = ~raveBus        ? 64;     // must match rave_layer.RAVE_BUS

// key <-> file. Keys are shared with rave_layer.MODELS (test_rave_layer.py
// checks the two lists agree); the GUI's home-row order is this order.
~raveModels = [
    [\birds,      "birds_dawnchorus_b2048_r48000_z8.ts"],
    [\crozzoli,   "crozzoli_bigensemblesmusic_18d.ts"],
    [\guitar,     "guitar_iil_b2048_r48000_z16.ts"],
    [\humpbacks,  "humpbacks_pondbrain_b2048_r48000_z20.ts"],
    [\magnets,    "magnets_b2048_r48000_z8.ts"],
    [\marine,     "marinemammals_pondbrain_b2048_r48000_z20.ts"],
    [\mrp,        "mrp_strengjavera_b2048_r44100_z16.ts"],
    [\organ,      "organ_archive_b2048_r48000_z16.ts"],
    [\voicemulti, "voice-multi-b2048-r48000-z11.ts"],
    [\vctk,       "voice_vctk_b2048_r44100_z22.ts"],
    [\vocalset,   "voice_vocalset_b2048_r48000_z16.ts"],
    [\water,      "water_pondbrain_b2048_r48000_z16.ts"]
];

~raveLoaded = List.new;          // keys whose SynthDef exists
~raveStatus = Dictionary.new;    // key -> (status: rate: minBufferSize: error:)

// Trained sample rate from the file name ("_r48000_" / "-r48000-"), nil if
// absent. Older nn.ar releases have no NNModelInfo.sampleRate accessor, so
// this is the fallback (and the only source for crozzoli, whose name has none).
~raveRateFromName = { |file|
    var m = file.findRegexp("[_-]r([0-9]+)[_-]");
    if (m.size >= 2) { m[1][1].asInteger } { nil };
};

s.waitForBoot({
    Routine({
        var nn = \NN.asClass;
        var statusPath = ~raveModelDir +/+ "rave_status.json";
        var jsonStr, writeStatus, srvRate;

        srvRate = s.sampleRate.asInteger;

        jsonStr = { |x| "\"" ++ x.asString.replace("\\", "/").replace("\"", "'").replace("\n", " ") ++ "\"" };

        writeStatus = {
            var f, entries;
            entries = ~raveModels.collect { |pair|
                var key = pair[0];
                var st  = ~raveStatus[key] ? (status: "unknown", error: "");
                "%: {\"status\": %, \"rate\": %, \"minBufferSize\": %, \"error\": %}".format(
                    jsonStr.(key), jsonStr.(st[\status]),
                    st[\rate] ? "null", st[\minBufferSize] ? "null", jsonStr.(st[\error] ? ""))
            };
            f = File(statusPath, "w");
            if (f.isOpen) {
                f.write("{\n  \"nn_available\": %,\n  \"sampleRate\": %,\n  \"bufferSize\": %,\n  \"written\": %,\n  \"models\": {\n    %\n  }\n}\n"
                    .format(nn.notNil, srvRate, ~raveBufferSize, jsonStr.(Date.getDate.stamp),
                            entries.join(",\n    ")));
                f.close;
                ("blob_synth_rave.scd: status written to " ++ statusPath).postln;
            } {
                ("blob_synth_rave.scd: could not write " ++ statusPath
                 ++ " (Python will list the models as unverified)").warn;
            };
        };

        ("blob_synth_rave.scd: model dir " ++ ~raveModelDir).postln;

        // ── 1. model descriptions ─────────────────────────────────────────
        if (nn.isNil) {
            ("blob_synth_rave.scd: nn.ar is NOT installed -- no RAVE SynthDefs. "
             ++ "Install the release from https://github.com/elgiano/nn.ar/releases, "
             ++ "then on macOS: NN.deQuarantine; s.reboot;").warn;
            ~raveModels.do { |pair|
                ~raveStatus[pair[0]] = (status: "failed", error: "nn.ar not installed");
            };
        } {
            ~raveModels.do { |pair|
                var key = pair[0], file = pair[1];
                var path = ~raveModelDir +/+ file;
                var model, info, rate, minBuf;
                if (File.exists(path).not) {
                    ~raveStatus[key] = (status: "missing", error: "file not found");
                    ("  - % : file missing (%)".format(key, path)).postln;
                } {
                    ("  . % : loading description ...".format(key)).postln;
                    try {
                        // NN.load inside a Routine waits for scsynth to answer.
                        // ALWAYS reload: sclang's model store survives a server
                        // reboot but scsynth's does not, so a SynthDef built
                        // from a cached description would reference a model id
                        // the server no longer has ("NNModelDescLib: id N not
                        // found. Loaded models: []") and the node is silent.
                        nn.load(key, path);
                        model = nn.model(key);
                        info  = model !? { model.info };
                        if (info.isNil) { Error("no model info after load").throw };
                        if (model.method(\forward).isNil) { Error("model has no 'forward' method").throw };
                        // accessors that only newer nn.ar releases have
                        rate   = info.tryPerform(\sampleRate) ?? { ~raveRateFromName.(file) };
                        minBuf = info.tryPerform(\minBufferSize);
                        ~raveStatus[key] = (status: "loaded", rate: rate,
                                            minBufferSize: minBuf, error: "");
                        ~raveLoaded.add(key);
                        ("  + % : loaded (rate % Hz, minBufferSize %)"
                         .format(key, rate ? "?", minBuf ? "model default")).postln;
                        if (rate.notNil and: { rate != srvRate }) {
                            ("    ! % was trained at % Hz but the server runs at % Hz: "
                             ++ "nn.ar does not resample, expect a slight spectral tilt"
                             ).format(key, rate, srvRate).postln;
                        };
                    } { |err|
                        ~raveStatus[key] = (status: "failed", error: err.errorString);
                        ("  ! % : FAILED -- %".format(key, err.errorString)).postln;
                    };
                };
            };
        };

        // ── 2. one SynthDef per loaded model ──────────────────────────────
        // amp     0..1   wet level (Python: mix / sqrt(active models))
        // inGain  0.05..8 gain into the model (RAVE likes a healthy level)
        // pan     -1..1  fixed per model (bank spread)
        // gate    1/0    0 = fade out over fadeOut s, then free (doneAction 2)
        ~raveLoaded.copy.do { |key|
            var name = ("raveXfer_" ++ key).asSymbol;
            var fwd  = nn.model(key).method(\forward);
            var nIn  = fwd.numInputs.max(1), nOut = fwd.numOutputs.max(1);
            try {
                SynthDef(name, {
                    |inBus = 64, out = 0, amp = 0.5, inGain = 1.0, pan = 0,
                     gate = 1, fadeIn = 3.0, fadeOut = 1.0|   // v4.11.6: 3 s covers model load
                    var in, g, ins, wet, env, sig;
                    in  = In.ar(inBus, 2);
                    g   = inGain.lag(0.2).clip(0.05, 8);
                    // most RAVE models are mono (1 in / 1 out): feed the L+R
                    // sum. A 2-input model gets L and R; anything else gets
                    // the sum on every input.
                    ins = if (nIn == 2) { [in[0], in[1]] * g } { ((in[0] + in[1]) * 0.5 * g) ! nIn };
                    wet = nn.new(key, \forward).ar(ins, ~raveBufferSize).asArray;
                    env = EnvGen.kr(Env.asr(fadeIn.max(0.01), 1, fadeOut.max(0.01), \sin),
                                    gate, doneAction: 2);
                    wet = Limiter.ar(LeakDC.ar(wet), 0.9, 0.01) * env * amp.lag(0.3).clip(0, 1);
                    sig = if (wet.size >= 2) {
                        Balance2.ar(wet[0], wet[1], pan.lag(0.5).clip(-1, 1))
                    } {
                        Pan2.ar(wet[0], pan.lag(0.5).clip(-1, 1))
                    };
                    Out.ar(out, sig);
                }).add;
                ("  + SynthDef % (forward: % in, % out)".format(name, nIn, nOut)).postln;
            } { |err|
                ~raveStatus[key] = (status: "failed", error: "SynthDef: " ++ err.errorString);
                ~raveLoaded.remove(key);
                ("  ! SynthDef % failed -- %".format(name, err.errorString)).postln;
            };
        };

        // ── 3. the split ──────────────────────────────────────────────────
        // Copies bus 0 to raveBus for the models and scales the dry mix.
        // dry = 1 -> transparent passthrough (layer off / no model active);
        // dry = 1 - mix while the layer is on. Lagged so toggling never clicks.
        //
        // v4.11.2: the copy sent to the models is SANITISED (NaN / inf ->
        // silence per channel, CheckBadValues) and LIMITED at 0.7. A RAVE
        // model fed a non-finite or wildly over-range sample corrupts its
        // cached convolution state and that instance stays silent or garbage
        // until it is freed -- one bad SynthDef upstream would otherwise
        // silence the whole layer. The dry path is untouched.
        SynthDef(\raveSplit, { |raveBus = 64, dry = 1.0|
            var in = In.ar(0, 2);
            // v4.11.3: also hard-clip at +/-1: a huge FINITE value (1e28)
            // passes CheckBadValues and overflows the Limiter into NaN
            var safe = in.collect { |ch|
                Select.ar(CheckBadValues.ar(ch, 0, 0) > 0, [ch, DC.ar(0)]).clip2(1.0)
            };
            Out.ar(raveBus, Limiter.ar(safe, 0.7, 0.01));
            ReplaceOut.ar(0, in * dry.lag(0.5).clip(0, 1));
        }).add;

        s.sync;

        // ── 4. status for Python ──────────────────────────────────────────
        writeStatus.();

        ("Audible Lava v4.11: RAVE layer ready -- % of % models loaded: %"
         .format(~raveLoaded.size, ~raveModels.size,
                 if (~raveLoaded.isEmpty) { "none" } { ~raveLoaded.join(" ") })).postln;
        if (nn.notNil and: { ~raveLoaded.isEmpty }) {
            ("  (no model loaded: check the paths above; NN.debug(true) before "
             ++ "re-evaluating prints the backend's reasons)").postln;
        };
    }).play(AppClock);
});


// ── Language-side helpers (manual tests / sessions without Python) ──────────
// In a live run Python owns these nodes (VoiceManager's /g_freeAll 0 at
// startup would free anything spawned here first).
~raveNodes = ~raveNodes ? ();
~raveOn = { |key, mix = 0.8|
    var name = ("raveXfer_" ++ key).asSymbol;
    if (~raveLoaded.includes(key).not) {
        ("raveOn: % is not loaded (loaded: %)".format(key, ~raveLoaded)).warn;
    } {
        ~raveSplit = ~raveSplit ?? { Synth.tail(s, \raveSplit, [\raveBus, ~raveBus, \dry, 1 - mix]) };
        ~raveSplit.set(\dry, 1 - mix);
        ~raveNodes[key] = Synth.tail(s, name, [\inBus, ~raveBus, \amp, mix, \inGain, 1.0]);
        ("raveOn: % (mix %)".format(key, mix)).postln;
    };
    ~raveNodes[key]
};
~raveOff = { |key|
    ~raveNodes[key] !? { |n| n.set(\gate, 0) };
    ~raveNodes.removeAt(key);
    if (~raveNodes.isEmpty) { ~raveSplit !? { |sp| sp.set(\dry, 1) } };
};
~raveAllOff = { ~raveNodes.keys.copy.do { |k| ~raveOff.(k) } };


// ── Manual tests (evaluate one block at a time after loading) ───────────────
/*

// which models made it, and what scsynth knows about one of them
~raveLoaded;
NN(\organ).describe;

// a sustained rich voice to transfer (needs blob_synth_rich.scd)
x = Synth(\mpeVoiceRich, [\freq, 110, \amp, 0.3, \cutoff, 1800, \res, 0.4, \wave, 0.5, \bank, 2]);

// hear it as an organ, then as birds on top; then dry again
~raveOn.(\organ, 0.9);          // a few seconds of silence while the model thread loads
~raveOn.(\birds, 0.9);
~raveNodes[\organ].set(\amp, 0.5);
~raveNodes[\birds].set(\inGain, 2.0);   // drive the model harder
~raveOff.(\birds);
~raveAllOff.();
x.set(\gate, 0);

// if nothing sounds: debug the backend and re-evaluate the file
NN.debug(true);

*/
