    // ── \blobStrike (v3.x; Klank transient on birth) ──────────────────────
    // v4.11.6: the Klank is struck by ONE impulse only. Before, it was
    // excited by Impulse + WhiteNoise.ar(0.01) running CONTINUOUSLY into five
    // resonators ringing up to 1.5 s; a resonator's gain on sustained input
    // grows with its ring time, so that quiet noise built up toward full
    // scale and smeared each birth into a metallic hiss (the same failure as
    // the first v4.11 \blobStrikeRich, at a smaller scale).
    // The breathy shimmer the noise gave is kept, OUTSIDE the resonators:
    // white noise band-passed at the same inharmonic partials, on a short
    // decaying envelope, so it shimmers with the ring but cannot accumulate.
    // Ratios, pitch, decay and the control contract are unchanged; a Limiter
    // (as \bassStutter has) is the ceiling.
    SynthDef(\blobStrike, {
        |freq=440, amp=0.4, decay=2.5, blobSize=0.5, position=0.5,
         elongation=0.0|
        var sig, env, freqMod, ratios, len, breath;
        ratios  = [1, 2.1, 3.3, 4.7, 6.2];
        freqMod = freq * (1.0 + elongation.clip(0, 1) * 0.12);
        len     = decay * blobSize + 0.1;
        env = EnvGen.kr(Env.perc(0.001, len), doneAction: 2);
        sig = Klank.ar(
            `[ratios, nil, [1.5, 0.8, 0.6, 0.4, 0.2]],
            Impulse.ar(0),
            freqMod
        );
        breath = Mix.new(
            BPF.ar(WhiteNoise.ar(1.0), (freqMod * ratios).clip(20, 18000), 0.03)
            * [0.5, 0.35, 0.25, 0.15, 0.1]
        ) * 4.0 * EnvGen.kr(Env.perc(0.005, (len * 0.5).clip(0.05, 1.5), 1, -4));
        sig = Limiter.ar(LeakDC.ar(sig + breath) * env * amp, 0.9);
        Out.ar(0, Pan2.ar(sig, position * 2 - 1));
    }).add;
