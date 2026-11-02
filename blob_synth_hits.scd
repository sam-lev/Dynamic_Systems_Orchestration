// blob_synth_hits.scd -- Audible Lava v4.2 (additive; evaluate after blob_synth.scd)
//
// \mpeHit  -- gated / percussive sibling of \mpeVoice for the pattern layer.
//             One node per HIT (self-freeing via the perc envelope). Timbre
//             family matches \mpeVoice -- same oscillator stack, same
//             RLPF/BPF/RHPF morph via SelectX, same tanh drive -- so hits and
//             sustained voices blend. `dur` sets the hit length, `vel` scales
//             amplitude and brightness. Scheduled from Python as timestamped
//             OSC bundles against the ElasticClock (never SC's TempoClock:
//             meter stays projective).
//
// \preSound -- very quiet, slow pad for the Imagination phase: the governor
//             pre-sounds the PREDICTED next triad at a gain equal to the
//             harmony posterior mass on non-current candidates. As that mass
//             grows the coming chord becomes audible-as-coming.
//
// NOTE: stock UGens only (no plugins). \mpeVoice's header comment says "SVF"
// but its code uses RLPF/BPF/RHPF + SelectX; this file follows the code.
//
(
SynthDef(\mpeHit, {
    |out    = 0,
     freq   = 220,
     amp    = 0.3,
     pan    = 0,
     dur    = 0.12,
     vel    = 0.8,
     cutoff = 2400,
     res    = 0.25,
     drive  = 0.1,
     noise  = 0.05,
     fmode  = 0.2,
     detune = 0.0|

    var env, sig, folded, air, lp, bp, hp, mix, f, cut, rq;

    // Percussive envelope; doneAction frees the node when the hit decays.
    env = EnvGen.kr(Env.perc(0.004, dur.clip(0.02, 8.0), 1.0, -4.0), doneAction: 2);

    // Slight per-hit detune wobble (matches the unison spread of \mpeVoice).
    f   = freq.clip(20, 12000) * (1.0 + (detune * 0.01 * LFNoise1.kr(7.0)));

    // Oscillator stack: saw core + octave sine + suboctave triangle.
    sig = Mix.new([
        VarSaw.ar(f,        0, (0.5 + (0.3 * vel)).clip(0.05, 0.95)),
        SinOsc.ar( f * 2.0, 0, 0.30 * (1.0 - fmode.clip(0.0, 1.0))),
        LFTri.ar(  f * 0.5, 0, 0.20)
    ]) * 0.5;

    // Drive: tanh wavefold crossfaded in (drive = 0 -> fully dry).
    folded = (sig * (1.0 + (drive.clip(0, 1) * 4.0))).tanh;
    sig    = XFade2.ar(sig, folded, (drive.clip(0, 1) * 2.0) - 1.0);

    // Filter morph LP <-> BP <-> HP, brightness tracking velocity.
    cut = (cutoff * (0.6 + (0.8 * vel))).clip(40, 18000);
    rq  = res.clip(0.05, 1.0).linlin(0.05, 1.0, 1.0, 0.05);
    lp  = RLPF.ar(sig, cut, rq);
    bp  = BPF.ar( sig, cut, rq);
    hp  = RHPF.ar(sig, cut, rq);
    mix = SelectX.ar(fmode.clip(0.0, 1.0) * 2.0, [lp, bp, hp]);

    // Short breath transient blended in by `noise`.
    air = BPF.ar(PinkNoise.ar(1.0), f.clip(60, 8000), 0.5) * 6.0
          * EnvGen.kr(Env.perc(0.001, (dur * 0.4).clip(0.01, 2.0)));
    mix = mix + (air * noise.clip(0, 1) * 0.5);

    Out.ar(out, Pan2.ar(mix * env * amp * (0.4 + (0.6 * vel)), pan.clip(-1, 1)));
}).add;


SynthDef(\preSound, {
    |out  = 0,
     freq = 220,
     amp  = 0.0,
     gate = 1,
     pan  = 0|

    var env, sig, ampL;

    env  = EnvGen.kr(Env.asr(2.0, 1.0, 3.0), gate, doneAction: 2);
    // Linear glide on amp so the governor can raise/lower the pre-sound
    // continuously as posterior mass shifts (same VarLag idiom as \mpeVoice).
    ampL = VarLag.kr(amp, 1.5, warp: \linear);

    sig = Mix.new([
        SinOsc.ar(freq,         0, 1.0),
        SinOsc.ar(freq * 1.005, 0, 0.5),
        SinOsc.ar(freq * 2.01,  0, 0.15)
    ]) * 0.3;
    sig = LPF.ar(sig, (freq * 3.0).clip(80, 12000));

    Out.ar(out, Pan2.ar(sig * env * ampL, pan.clip(-1, 1)));
}).add;

"Audible Lava v4.2: pattern-layer SynthDefs ready (mpeHit, preSound)".postln;
)

/*
Quick tests (evaluate one line at a time, server booted):

Synth(\mpeHit, [\freq, 330, \amp, 0.4, \dur, 0.15, \vel, 0.9]);
Synth(\mpeHit, [\freq, 110, \amp, 0.4, \dur, 0.5, \vel, 0.4, \fmode, 0.0, \drive, 0.6]);

// a quick burst to hear the hit family against \mpeVoice
Routine({ 8.do { |i|
    Synth(\mpeHit, [\freq, 220 * [1, 1.5, 2, 2.5].wrapAt(i), \amp, 0.3,
                    \dur, 0.12, \vel, [0.9, 0.5].wrapAt(i)]);
    0.15.wait;
} }).play;

p = Synth(\preSound, [\freq, 196, \amp, 0.06]);
p.set(\amp, 0.12);   // posterior mass grows -> pre-sound rises
p.set(\gate, 0);     // release
*/
