// melody_hit.scd -- Audible Lava / DSO: one-shot melody note for melody_rhythm.py
//
// ADDITIVE. Evaluate this file AFTER blob_synth.scd (it needs ~wtBuffers for
// the wavetable bank). Nothing in blob_synth.scd changes. If you prefer a
// single file, paste the SynthDef block into blob_synth.scd's
// s.waitForBoot { ... } body, after \bassStutter and before s.sync.
//
// Spawned by melody_rhythm.MelodyEngine via
//     /s_new melodyHit -1 0 0 freq .. amp .. pan .. dur .. attack .. release ..
//            cutoff .. res .. fmode .. wave .. bank .. detune .. drive ..
// Node id -1 = server-assigned; the envelope frees the node (doneAction: 2),
// so Python never needs a note-off.
//
// Timbre mirrors \mpeVoice (same wavetable bank index, 3-osc unison detune,
// LP/BP/HP morph, tanh wavefold) so a blob's melody notes sound like its drone.
// Envelope: attack -> hold for `dur` -> `release` (Env.linen).

(
s.waitForBoot {
    SynthDef(\melodyHit, {
        |out     = 0,
         freq    = 220,
         amp     = 0.2,
         pan     = 0,
         dur     = 0.5,      // hold time, seconds (Python: 0.25 + 2.5*sqrt(area_n))
         attack  = 0.02,
         release = 0.6,      // 1.2 legato (expanding) / 0.25 staccato (shrinking)
         cutoff  = 1500,
         res     = 0.3,
         fmode   = 0.0,      // 0=LP, 0.5=BP, 1=HP
         wave    = 0,        // 0..3 within bank (continuous)
         bank    = 0,        // 0..3
         bufBase = 0,
         detune  = 0.0,
         drive   = 0.0|

        var env, idx, spread, oscs, sig, lp, bp, hp, mix, cutL, resL;

        env = EnvGen.kr(
            Env.linen(attack.max(0.002), dur.max(0.02), release.max(0.02),
                      1.0, \sin),
            doneAction: 2
        );

        idx    = bufBase + (bank.clip(0, 3) * 4) + wave.clip(0, 3.999);
        spread = detune.clip(0, 1) * 0.04;
        oscs   = VOsc.ar(idx, freq * [1.0, 1.0 + spread, 1.0 - spread]);
        sig    = oscs.sum * (1.0 / 3);

        // Wavefold-ish saturation, unity at drive = 0
        sig = (sig * (1.0 + (drive.clip(0, 1) * 6.0))).tanh
              / (1.0 + (drive.clip(0, 1) * 0.5));

        // Filter morph (a cheap stand-in for the SVF in \mpeVoice)
        cutL = cutoff.clip(40, 16000);
        resL = res.clip(0.05, 1.0);
        lp   = RLPF.ar(sig, cutL, 1.0 - (resL * 0.9));
        bp   = BPF.ar(sig,  cutL, (1.0 - resL).max(0.1)) * 2.0;
        hp   = RHPF.ar(sig, cutL, 1.0 - (resL * 0.9));
        mix  = SelectX.ar(fmode.clip(0, 1) * 2, [lp, bp, hp]);

        // Short, bright transient so the onset reads as a *note* against the
        // slow drone underneath it.
        mix = mix + (HPF.ar(sig, 2000) * EnvGen.kr(Env.perc(0.001, 0.06)) * 0.35);

        Out.ar(out, Pan2.ar(LeakDC.ar(mix) * env * amp.clip(0, 0.8), pan.clip(-1, 1)));
    }).add;

    s.sync;
    "melody_hit.scd: \\melodyHit ready".postln;
};
)

// ── Manual test (evaluate after loading; one line at a time) ────────────────
/*
Synth(\melodyHit, [\freq, 330, \amp, 0.3, \dur, 0.6, \release, 1.2, \cutoff, 2500]);
Synth(\melodyHit, [\freq, 440, \amp, 0.3, \dur, 0.2, \release, 0.25, \drive, 0.6, \fmode, 0.5]);
// Fake a hazard burst: 6 notes stepping up the pentatonic
(
Routine({
    [48, 50, 52, 55, 57, 60].do { |m|
        Synth(\melodyHit, [\freq, m.midicps, \amp, 0.25, \dur, 0.3, \release, 0.4]);
        0.35.wait;
    };
}).play;
)
*/
