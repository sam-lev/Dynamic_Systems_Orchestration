// rave_bench.scd -- Audible Lava / DSO v4.11 RAVE test bench (SuperCollider half)
//
// Evaluate AFTER blob_synth_rave.scd (needs the model descriptions nn.ar has
// loaded, the rich SynthDefs and ~has/~ug from blob_synth_rich.scd). Then run
//   python rave_bench.py            # full automated sweep -> WAVs + report
//   python rave_bench.py --listen   # play each scenario, no recording
// (see rave_bench.py for the scenario list and flags).
//
// WHAT THIS FILE ADDS
//
//   \raveBench_<key>_<buf>   one SynthDef per loaded model per buffer size in
//                            ~benchBuffers, identical to \raveXfer_<key> but
//                            with the buffer size baked in. This is how the
//                            bench sweeps buffer size without re-evaluating
//                            blob_synth_rave.scd: the live layer keeps using
//                            \raveXfer_<key> (~raveBufferSize); the bench
//                            spawns these instead.
//   \benchBurst              latency probe: a 60 ms pink-noise burst every
//                            `period` s, hard-gated so its onsets are sharp
//                            enough to cross-correlate dry vs wet.
//   \benchTone               a steady band-limited tone (for level / silence
//                            / "does the model hallucinate" checks).
//   \benchRec                records bus 0 (stereo) to disk through buffer
//                            `buf` (DiskOut); Python opens/closes the file.
//   \benchMeter              posts peak / RMS of bus 0 twice a second to the
//                            post window while a scenario plays (listen mode).
//
// Everything is spawned by Python at the HEAD of group 0 (stimuli) or BEFORE
// the master (RAVE nodes), exactly as the live system does, so the bench
// exercises the real node order: stimulus -> \raveSplit -> models ->
// \lavaMaster (if Python chose to spawn it) -> \benchRec reads bus 0 at the
// very tail.
//
// Every nn.ar reference goes through \NN.asClass (compiles without it).

~benchBuffers = ~benchBuffers ? [2048, 4096, 8192];
~raveBus      = ~raveBus ? 64;

s.waitForBoot({
    Routine({
        var nn = \NN.asClass;
        var built = List.new;

        if (~raveLoaded.isNil or: { ~raveLoaded.isEmpty }) {
            "rave_bench.scd: no RAVE models loaded -- evaluate blob_synth_rave.scd first".warn;
        };

        // ── per-model, per-buffer SynthDefs ───────────────────────────────
        (~raveLoaded ? []).do { |key|
            var fwd  = nn.model(key).method(\forward);
            var nIn  = fwd.numInputs.max(1), nOut = fwd.numOutputs.max(1);
            ~benchBuffers.do { |buf|
                var name = ("raveBench_" ++ key ++ "_" ++ buf).asSymbol;
                try {
                    SynthDef(name, {
                        |inBus = 64, out = 0, amp = 0.5, inGain = 1.0, pan = 0,
                         gate = 1, fadeIn = 0.05, fadeOut = 0.3|
                        var in, g, ins, wet, env, sig;
                        in  = In.ar(inBus, 2);
                        g   = inGain.lag(0.2).clip(0.05, 8);
                        ins = if (nIn == 2) { [in[0], in[1]] * g } { ((in[0] + in[1]) * 0.5 * g) ! nIn };
                        wet = nn.new(key, \forward).ar(ins, buf).asArray;
                        env = EnvGen.kr(Env.asr(fadeIn.max(0.01), 1, fadeOut.max(0.01), \sin),
                                        gate, doneAction: 2);
                        wet = Limiter.ar(LeakDC.ar(wet), 0.9, 0.01) * env * amp.lag(0.1).clip(0, 1);
                        sig = if (wet.size >= 2) {
                            Balance2.ar(wet[0], wet[1], pan.lag(0.5).clip(-1, 1))
                        } { Pan2.ar(wet[0], pan.lag(0.5).clip(-1, 1)) };
                        Out.ar(out, sig);
                    }).add;
                    built.add(name);
                } { |err|
                    ("  ! % failed -- %".format(name, err.errorString)).postln;
                };
            };
        };

        // ── stimuli ───────────────────────────────────────────────────────
        // latency probe: hard-gated pink bursts on an APERIODIC schedule
        // (period * [0.7 1.3 0.9 1.1 0.8 1.2]) so the dry/wet cross-correlation
        // has one unambiguous peak; mono-centred, no reverb tail
        // gate: 0 frees the node (it ignored gate before and kept running
        // under every later scenario -- the "tsk tsk ... tsk tsk")
        SynthDef(\benchBurst, { |out = 0, amp = 0.5, period = 1.0, width = 0.06, freq = 0, gate = 1|
            var trig, env, src;
            EnvGen.kr(Env.asr(0, 1, 0.01), gate, doneAction: 2);
            trig = TDuty.kr(Dseq([0.7, 1.3, 0.9, 1.1, 0.8, 1.2] * period.max(0.1), inf));
            env  = EnvGen.kr(Env([0, 1, 1, 0], [0.002, width, 0.004]), trig);
            // freq is a control (a UGen), so choose at signal rate, not with `if`
            src  = Select.ar(freq > 0, [PinkNoise.ar(1.0),
                                        (Saw.ar(freq.max(20)) * 0.5) + PinkNoise.ar(0.5)]);
            src  = LPF.ar(src, 6000);
            Out.ar(out, (src * env * amp) ! 2);
        }).add;

        // steady tone (freq 0 = pink noise) for level / hallucination checks
        SynthDef(\benchTone, { |out = 0, amp = 0.3, freq = 220, gate = 1, pan = 0|
            var env, src;
            env = EnvGen.kr(Env.asr(0.05, 1, 0.1), gate, doneAction: 2);
            // freq 0 = pink noise; chosen at signal rate (freq is a control)
            src = Select.ar(freq > 0, [
                PinkNoise.ar(0.8),
                (VarSaw.ar(freq.max(20), 0, 0.4) + SinOsc.ar(freq.max(20) * 2, 0, 0.3)
                 + LFTri.ar(freq.max(20) * 0.5, 0, 0.2)) * 0.5
            ]);
            src = LPF.ar(src, 7000);
            Out.ar(out, Pan2.ar(src * env * amp, pan));
        }).add;

        // recorder: tail of group 0, reads the finished stereo mix
        SynthDef(\benchRec, { |buf = 0|
            DiskOut.ar(buf, In.ar(0, 2));
        }).add;

        // meter for listen mode
        SynthDef(\benchMeter, { |rate = 2|
            var in = In.ar(0, 2);
            var pk = Peak.ar(in, Impulse.kr(rate)).max(1e-6);
            var rms = (RunningSum.ar(in.squared, 2048) / 2048).sqrt.max(1e-6);
            SendReply.kr(Impulse.kr(rate), "/benchMeter",
                         [pk[0].ampdb, pk[1].ampdb, rms[0].ampdb, rms[1].ampdb]);
        }).add;

        s.sync;

        // listen-mode post of the meter (Python only reads WAVs)
        OSCdef(\benchMeter, { |msg|
            ("  bus0  peak L % R %   rms L % R %  dB"
             .format(msg[3].round(0.1), msg[4].round(0.1), msg[5].round(0.1), msg[6].round(0.1))).postln;
        }, "/benchMeter");

        ("rave_bench.scd: % bench SynthDefs built for buffers % (models: %)"
         .format(built.size, ~benchBuffers, (~raveLoaded ? []).join(" "))).postln;
        "rave_bench.scd: stimuli ready (benchBurst benchTone benchRec benchMeter). Now run rave_bench.py".postln;
    }).play(AppClock);
});
