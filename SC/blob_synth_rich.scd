// blob_synth_rich.scd -- Audible Lava / DSO v4.10 "Rich Synth Palette"
//
// v4.11 (Nov 2026): \blobStrikeRich is rewritten as an orchestral swell
// (see its header below; v4.11.2 removes a resonator overload).
// v4.11.3: ~safe output guard (NaN/inf scrub + ceiling) on every SynthDef
// with a feedback loop, and on \\lavaMaster's input.
//
// ADDITIVE. Evaluate AFTER blob_synth.scd (it needs the 16 wavetable buffers
// at ids 0..15). blob_synth_hits.scd and melody_hit.scd may still be loaded
// for the classic palette; nothing in them is touched. Python selects which
// palette sounds through synth_palette.py (apply_palette / install), so the
// classic SynthDef names keep working unchanged and `--classic-synths`
// falls back to them.
//
// WHAT THIS FILE ADDS
//
//   Drop-in siblings. Each rich SynthDef accepts EXACTLY the control names
//   the Python side already sends to its classic twin (so no /s_new or /n_set
//   call has to change) plus a few optional extras that default to 0 /
//   "auto". The classic twin's argument list is the contract:
//
//     \mpeVoiceRich    <- \mpeVoice     freq amp pan gate attack decay sustain
//                                       release cutoff res fmode wave bank
//                                       bufBase detune shimmer noise drive
//                                       + sub, formant, chorus (extras)
//     \mpeHitRich      <- \mpeHit       freq amp pan dur vel cutoff res drive
//                                       noise fmode detune
//                                       + model (0 wavetable, 1 plucked, 2 piano)
//     \melodyHitRich   <- \melodyHit    freq amp pan dur attack release cutoff
//                                       res fmode wave bank bufBase detune drive
//                                       + model (-1 auto, 0 wt, 1 plucked, 2 bowed), bow
//     \bassStutterRich <- \bassStutter  freq amp pan gate stutterRate lfoDepth
//                                       + crush
//     \blobStrikeRich  <- \blobStrike   freq amp decay blobSize position elongation
//     \blobAngleRich   <- \blobAngle    freq amp gate orientation attack release
//     \preSoundRich    <- \preSound     freq amp gate pan
//
//   Master space.  \lavaMaster sits at the TAIL of group 0, reads the mix on
//   bus 0 (every voice, hit, strike and pad already writes there) and
//   ReplaceOut's the processed stereo image. No Python routing changes: node
//   order is execution order, Python spawns everything at the HEAD of group 0
//   (/s_new ... 0 0), so a tail node always sees the complete sum. Python's
//   synth_palette.MasterSpace spawns it (node 900) AFTER VoiceManager's
//   /g_freeAll 0 and drives its parameters every frame from ScoreState:
//   ambiguity -> reverb time, tension -> damping/saturation, phrase position
//   -> wet mix, surprisal -> Greyhole feedback, Kuramoto order -> modulation
//   depth, blob count -> room size, grouping boundary -> bloom. The language
//   side can also spawn it by hand for tests (~spawnMaster below).
//
//   \lavaMasterVST is the same insert built around VSTPlugin.ar so a VST3
//   (e.g. ValhallaSupermassive) can be the space instead; its params are
//   exposed as p0..p7 so Python can /n_set them like any other control.
//   Off unless VSTPlugin is installed and ~useVST is true.
//
// PLUGIN DEPENDENCIES (sc3-plugins 3.13 release, all optional)
//
//   Every plugin UGen is reached through \Name.asClass so this file COMPILES
//   without sc3-plugins (a literal class name that does not exist is a
//   compile error for the whole file). Each use checks ~has.(\Name) at graph
//   build time and substitutes stock UGens when the plugin is absent, so the
//   rich palette degrades to roughly the classic sound rather than failing.
//   The post window lists which plugins were found.
//
//     SVF           BlackrainUGens  true state-variable filter with LP/BP/HP outputs
//     MoogLadder    MoogLadderUGens 4-pole ladder for the bass voice
//     DWGPlucked    DWGUGens        digital waveguide plucked string
//     DWGBowed      DWGUGens        digital waveguide bowed string
//     OteyPiano     OteyPianoUGens  physical piano (fallback MdaPiano, then Klank)
//     MembraneCircle MembraneUGens  2-D waveguide drum for births
//     Streson       BhobUGens       string resonator ("air" that rings)
//     Gendy1        (core)          Xenakis dynamic stochastic synthesis for the pad
//     BlitB3Square/Saw AntiAliasingOscillators  band-limited sub oscillators
//     SmoothDecimator BerlachUGens  smoothed bit/rate crusher for the bass
//     JPverb, Greyhole DEINDUGens   the master space
//     VSTPlugin     separate extension (IEM) -- optional master
//
// Design notes on the mappings live in synth_palette.py and in the v4.10
// companion PDF; this file only documents what each control does.

s.options.sampleRate          = 48000;
s.options.numInputBusChannels = 0;
s.boot;

s.waitForBoot({
    var found, missing;

    ~bufBase = ~bufBase ? 0;
    ~useVST  = ~useVST  ? false;

    // ── plugin presence ───────────────────────────────────────────────────
    ~has = { |sym| sym.asClass.notNil };
    ~ug  = { |sym| sym.asClass };          // nil when absent; checked first

    found = List.new; missing = List.new;
    [\SVF, \MoogLadder, \DWGPlucked, \DWGBowed, \OteyPiano, \MdaPiano,
     \MembraneCircle, \Streson, \BlitB3Square, \BlitB3Saw, \SmoothDecimator,
     \JPverb, \Greyhole, \VSTPlugin].do { |sym|
        if (~has.(sym)) { found.add(sym) } { missing.add(sym) };
    };
    ("blob_synth_rich.scd: plugins found   " ++ found.join(" ")).postln;
    if (missing.notEmpty) {
        ("blob_synth_rich.scd: plugins MISSING " ++ missing.join(" ")
         ++ "  (stock fallbacks will be used)").postln;
    };

    // ── shared building blocks ────────────────────────────────────────────
    // Filter morph: one SVF pass when available (LP/BP/HP outputs weighted by
    // fmode), else the classic RLPF/BPF/RHPF + SelectX. `res` is 0..1 in
    // both; SVF takes it directly, the stock trio gets rq = 1 - 0.95 res.
    ~morphFilter = { |sig, cutoff, res, fmode|
        var lpW, bpW, hpW, m, lp, bp, hp, rq;
        m   = fmode.clip(0, 1) * 2;
        lpW = (1 - m).clip(0, 1);
        bpW = (1 - (m - 1).abs).clip(0, 1);
        hpW = (m - 1).clip(0, 1);
        if (~has.(\SVF)) {
            ~ug.(\SVF).ar(sig, cutoff.clip(30, 18000), res.clip(0, 0.98),
                          lpW, bpW, hpW, 0, 0)
        } {
            rq = res.clip(0.05, 1.0).linlin(0.05, 1.0, 1.0, 0.05);
            lp = RLPF.ar(sig, cutoff, rq);
            bp = BPF.ar(sig,  cutoff, rq) * 1.5;
            hp = RHPF.ar(sig, cutoff, rq);
            SelectX.ar(m, [lp, bp, hp])
        };
    };

    // "Air" that rings: Streson is a string resonator; fed with pink noise at
    // the fundamental it sounds like breath exciting a string rather than
    // filtered hiss. Fallback is the classic band-passed noise.
    ~airResonator = { |freq, res = 0.9|
        var f = freq.clip(50, 6000);
        if (~has.(\Streson)) {
            ~ug.(\Streson).ar(PinkNoise.ar(0.25), 1 / f, res.clip(0.5, 0.995))
        } {
            BPF.ar(PinkNoise.ar(1.0), f, 0.5) * 6.0
        };
    };

    // v4.11.3: output guard for anything with a feedback loop (waveguides,
    // physical models, resonators). NaN / inf -> 0 per channel, then a hard
    // ceiling at +/-ceil. A runaway voice is then an ugly clipped note
    // instead of a value that grows to 1e28, overflows the master Limiter,
    // and silences the entire mix (the bench measured +568 dBFS).
    ~safe = { |sig, ceil = 1.0|
        sig.asArray.collect { |ch|
            Select.ar(CheckBadValues.ar(ch, 0, 0) > 0, [ch, DC.ar(0)]).clip2(ceil)
        }.unbubble
    };

    ~bandSub = { |freq, square = 1|
        var f = freq.clip(16, 4000);
        if (~has.(\BlitB3Square) and: { square > 0 }) {
            ~ug.(\BlitB3Square).ar(f)
        } {
            if (~has.(\BlitB3Saw) and: { square <= 0 }) {
                ~ug.(\BlitB3Saw).ar(f)
            } {
                if (square > 0) { Pulse.ar(f, 0.5) } { Saw.ar(f) }
            }
        };
    };

    // ── \mpeVoiceRich ─────────────────────────────────────────────────────
    // The sustained blob voice. Everything \mpeVoice does, plus:
    //   * true SVF morph (one pass, no phase smear between LP/BP/HP branches)
    //   * `noise` becomes a ringing string-resonator "air" (Streson) --
    //     dissolving topology (D_PERSISTENCE < 0, AMBIGUITY) now reads as
    //     breath on a string, not hiss
    //   * `shimmer` adds a chorus (two modulated delays) on top of the
    //     vibrato/tremolo, so turning/acceleration widens the image
    //   * `drive` is a soft wavefold THEN tanh, so TENSION/D_AREA -> DRIVE
    //     grows odd harmonics before it saturates
    //   * extras (default off): sub = band-limited sub-octave blend,
    //     formant = vowel tilt (0..1 maps 'oh' -> 'ah' -> 'ee') driven by
    //     height via synth_palette, chorus = extra width independent of shimmer
    // All extras at 0 and the plugins missing gives a sound within a hair of
    // the classic voice, so switching palettes mid-session is safe.
    SynthDef(\mpeVoiceRich, {
        |out = 0, freq = 220, amp = 0.0, pan = 0, gate = 1,
         attack = 1.5, decay = 0.4, sustain = 0.7, release = 2.5,
         cutoff = 1500, res = 0.30, fmode = 0.0,
         wave = 0, bank = 0, bufBase = 0,
         detune = 0.0, shimmer = 0.0, noise = 0.0, drive = 0.0,
         sub = 0.0, formant = 0.0, chorus = 0.0|

        var env, idx, freqL, ampL, cutL, resL, panL, waveL;
        var detL, shimL, noiL, drvL, subL, forL, choL;
        var vib, trem, spread, oscs, sig, folded, air, wide, f1, f2, fmix, subSig;

        freqL = VarLag.kr(freq, 0.05, warp: \linear);
        ampL  = amp.lag(0.10);
        cutL  = cutoff.lag(0.05).clip(40, 16000);
        resL  = res.lag(0.05).clip(0.0, 1.0);
        panL  = pan.lag(0.05).clip(-1, 1);
        waveL = wave.lag(0.05).clip(0, 3.999);
        detL  = detune.lag(0.10).clip(0, 1);
        shimL = shimmer.lag(0.10).clip(0, 1);
        noiL  = noise.lag(0.10).clip(0, 1);
        drvL  = drive.lag(0.10).clip(0, 1);
        subL  = sub.lag(0.20).clip(0, 1);
        forL  = formant.lag(0.30).clip(0, 1);
        choL  = chorus.lag(0.20).clip(0, 1);

        idx = bufBase + (bank.clip(0, 3) * 4) + waveL;

        env = EnvGen.kr(
            Env.adsr(attack.lag(1.0).max(0.005), decay.max(0.005),
                     sustain.clip(0, 1), release.lag(1.0).max(0.005), curve: \sin),
            gate, doneAction: 2);

        // shimmer: vibrato + tremolo (as classic) ...
        vib  = SinOsc.kr(5.2 + (shimL * 1.5)).range(1 - (shimL * 0.03), 1 + (shimL * 0.03));
        trem = SinOsc.kr(5.2, 0.5pi).range(1 - (shimL * 0.35), 1);

        spread = detL * 0.04;
        oscs = [
            VOsc.ar(idx, freqL * vib),
            VOsc.ar(idx, freqL * vib * (1 - spread)),
            VOsc.ar(idx, freqL * vib * (1 + spread))
        ];
        sig = Mix.new(oscs) / 3;

        // band-limited sub an octave down (extra; off at 0)
        subSig = ~bandSub.(freqL * 0.5, 1) * 0.6;
        sig    = sig + (subSig * subL);

        // drive: fold first (odd harmonics), then tanh (saturation)
        folded = (sig * (1 + (drvL * 3))).fold2(1.0);
        folded = (folded * (1 + (drvL * 2))).tanh;
        sig    = XFade2.ar(sig, folded, (drvL * 2) - 1);

        // vowel tilt (extra): two parallel resonances sliding oh -> ah -> ee
        f1 = forL.linexp(0, 1, 420, 300).lag(0.3);     // F1 falls toward 'ee'
        f2 = forL.linexp(0, 1, 800, 2300).lag(0.3);    // F2 rises  toward 'ee'
        fmix = (BPF.ar(sig, f1, 0.25) + (BPF.ar(sig, f2, 0.2) * 0.7)) * 2.5;
        // blend in over formant 0..0.5, then only the vowel moves (0.5..1)
        sig  = XFade2.ar(sig, fmix, ((forL * 2).clip(0, 1) * 2) - 1);

        sig = ~morphFilter.(sig, cutL, resL, fmode.lag(0.05));

        // air: ringing string resonator at the fundamental
        air = ~airResonator.(freqL, 0.90);
        sig = sig + (air * noiL * 0.5);

        // ... and a chorus for width (shimmer + chorus extra)
        wide = (shimL * 0.6) + choL;
        sig  = [
            sig + (DelayC.ar(sig, 0.05, SinOsc.kr(0.31, 0).range(0.008, 0.016)) * wide * 0.6),
            sig + (DelayC.ar(sig, 0.05, SinOsc.kr(0.37, 2).range(0.010, 0.019)) * wide * 0.6)
        ] / (1 + (wide * 0.6));

        sig = ~safe.(LeakDC.ar(sig)) * env * ampL * trem;
        Out.ar(out, Balance2.ar(sig[0], sig[1], panL));
    }).add;

    // ── \mpeHitRich ───────────────────────────────────────────────────────
    // One node per hit (self-freeing). `model` selects the excitation:
    //   0  wavetable/VarSaw stack (classic timbre family)
    //   1  DWGPlucked -- a plucked string; vel -> pluck position/brightness,
    //      dur -> damping. Used for the ORNAMENTAL hit layer (low prominence).
    //   2  OteyPiano / MdaPiano hammer -- used for the PAIRED head voice an
    //      octave below the sustained note; vel -> hammer velocity.
    // The governor's extra params (cutoff/drive/noise/fmode/detune) still
    // shape all three: brightness, saturation and air are applied after the
    // excitation so tension and roughness read the same way across models.
    SynthDef(\mpeHitRich, {
        |out = 0, freq = 220, amp = 0.3, pan = 0, dur = 0.12, vel = 0.8,
         cutoff = 2400, res = 0.25, drive = 0.1, noise = 0.05, fmode = 0.2,
         detune = 0.0, model = 0|

        var env, f, src, wt, plk, pno, sig, folded, air, cut, d;

        d   = dur.clip(0.02, 8.0);
        env = EnvGen.kr(Env.perc(0.004, d, 1.0, -4.0), doneAction: 2);
        f   = freq.clip(20, 12000) * (1 + (detune * 0.01 * LFNoise1.kr(7.0)));

        // model 0: classic stack
        wt = Mix.new([
            VarSaw.ar(f, 0, (0.5 + (0.3 * vel)).clip(0.05, 0.95)),
            SinOsc.ar(f * 2, 0, 0.30 * (1 - fmode.clip(0, 1))),
            LFTri.ar(f * 0.5, 0, 0.20)
        ]) * 0.5;

        // model 1: plucked string
        // v4.11.5: DWGPlucked is excited ONLY by `inp` (the pluck itself:
        // a few ms of clipped noise, as in its help file). It was left at 0,
        // so the string never sounded and only faint ~9 kHz residue came out.
        // c3 is a high-frequency LOSS factor (higher = darker): 250-900 here,
        // velocity brightens by lowering it.
        plk = if (~has.(\DWGPlucked)) {
            var plk;
            plk = ~ug.(\DWGPlucked).ar(f, amp: 0.5 + (0.4 * vel), gate: 1,
                pos: 0.16 + (0.3 * (1 - vel)),         // nearer the middle = rounder (v4.11.6: +0.04)
                c1: (1 / d.max(0.05)).clip(0.5, 40),   // shorter hit = faster damping
                c3: 1300 - (600 * vel),                // HF loss 760-1000: darker (v4.11.6, was 315-575)
                inp: (0.5 + (0.4 * vel)) * LFClipNoise.ar(2000)
                     * EnvGen.ar(Env([0, 1, 1, 0], [0.001, 0.006, 0.0005], [5, -5, -8])),
                release: 0.05) * 0.15;
            // v4.11.6: gentle top roll-off, ~12th harmonic, so the ornament
            // sits under the drones instead of fizzing above them
            LPF.ar(plk, (f * 12).clip(1800, 7000))
        } {
            Pluck.ar(WhiteNoise.ar(0.4), Impulse.ar(0), 0.05, 1 / f, d * 1.5,
                     0.3 + (0.4 * (1 - vel)))
        };

        // model 2: piano hammer
        pno = if (~has.(\OteyPiano)) {
            ~ug.(\OteyPiano).ar(f, vel: 20 + (90 * vel), rho: 0.7 + (0.6 * vel)) * 1.5
        } {
            if (~has.(\MdaPiano)) {
                ~ug.(\MdaPiano).ar(f, gate: 1, vel: 20 + (100 * vel), decay: 0.6,
                                   release: 0.4, hard: vel, stereo: 0) * 1.5
            } {
                Klank.ar(`[[1, 2.01, 3.0, 4.2, 5.4], [1, 0.5, 0.3, 0.2, 0.1],
                           [1.2, 0.8, 0.5, 0.4, 0.3]], Impulse.ar(0) * 0.6, f)
            }
        };

        src = Select.ar(model.clip(0, 2), [wt, plk, pno]);

        folded = (src * (1 + (drive.clip(0, 1) * 4))).tanh;
        sig    = XFade2.ar(src, folded, (drive.clip(0, 1) * 2) - 1);

        cut = (cutoff * (0.6 + (0.8 * vel))).clip(40, 18000);
        sig = ~morphFilter.(sig, cut, res, fmode);

        air = ~airResonator.(f, 0.85) * EnvGen.kr(Env.perc(0.001, (d * 0.4).clip(0.01, 2)));
        sig = sig + (air * noise.clip(0, 1) * 0.5);

        sig = ~safe.(LeakDC.ar(sig));
        Out.ar(out, Pan2.ar(sig * env * amp * (0.4 + (0.6 * vel)), pan.clip(-1, 1)));
    }).add;

    // ── \melodyHitRich ────────────────────────────────────────────────────
    // melody_rhythm's one-shot note. `model`:
    //   -1 auto: release >= 0.8 (expanding blob, legato) -> bowed string;
    //            otherwise (shrinking, staccato)            -> plucked string
    //    0 wavetable (classic), 1 plucked, 2 bowed
    // `bow` (0..1) is bow pressure/force for the bowed model -- synth_palette
    // sends blob speed, so a fast rising blob is bowed harder (gritty) and a
    // slow one is drawn gently. The timbre controls (cutoff/res/fmode/
    // detune/drive) are still copied from the bound \mpeVoice so the note
    // keeps belonging to its blob.
    SynthDef(\melodyHitRich, {
        |out = 0, freq = 220, amp = 0.2, pan = 0, dur = 0.5, attack = 0.02,
         release = 0.6, cutoff = 1500, res = 0.3, fmode = 0.0, wave = 0, bank = 0,
         bufBase = 0, detune = 0.0, drive = 0.0, model = -1, bow = 0.3|

        var env, gateEnv, idx, spread, wt, plk, bwd, src, sig, m, d, hold;

        d    = dur.max(0.02);
        hold = attack.max(0.002) + d;
        env  = EnvGen.kr(Env.linen(attack.max(0.002), d, release.max(0.02), 1.0, \sin),
                         doneAction: 2);
        // a gate that closes when the hold ends, for the physical models
        gateEnv = EnvGen.kr(Env([1, 1, 0], [hold, 0.0]));

        // auto: legato release -> bowed (2), staccato -> plucked (1)
        m = Select.kr(model < 0, [model.clip(0, 2), (release >= 0.8) + 1]);

        idx    = bufBase + (bank.clip(0, 3) * 4) + wave.clip(0, 3.999);
        spread = detune.clip(0, 1) * 0.04;
        wt     = VOsc.ar(idx, freq * [1.0, 1 + spread, 1 - spread]).sum / 3;

        // v4.11.5: `inp` is the pluck (it was 0: no excitation, only faint
        // high residue); c3 is HF loss, 300-900 so the note has a body
        plk = if (~has.(\DWGPlucked)) {
            ~ug.(\DWGPlucked).ar(freq, amp: 0.7, gate: gateEnv, pos: 0.14 + (0.2 * bow),
                c1: (1 / (d + release).max(0.1)).clip(0.3, 20), c3: 900 - (600 * bow),
                inp: 0.7 * LFClipNoise.ar(2000)
                     * EnvGen.ar(Env([0, 1, 1, 0], [0.001, 0.006, 0.0005], [5, -5, -8])),
                release: release.max(0.05)) * 0.15
        } {
            Pluck.ar(WhiteNoise.ar(0.4), Impulse.ar(0), 0.05, 1 / freq.clip(20, 8000),
                     (d + release) * 1.2, 0.35)
        };

        bwd = if (~has.(\DWGBowed)) {
            ~ug.(\DWGBowed).ar(freq, velb: 0.3 + (0.4 * bow), force: 0.6 + (0.5 * bow),
                gate: gateEnv, pos: 0.12 + (0.1 * bow), release: release.max(0.05),
                c1: 0.25, c3: 10 + (20 * bow)) * 2.0
        } {
            // stock stand-in: saw through a slowly swelling resonant filter
            RLPF.ar(Saw.ar(freq * [1, 1.003]).sum * 0.5,
                    (freq * (2 + (4 * bow))).clip(100, 8000), 0.3)
            * EnvGen.kr(Env([0, 1, 1, 0], [0.15, hold, release.max(0.05)]))
        };

        src = Select.ar(m, [wt, plk, bwd]);

        sig = (src * (1 + (drive.clip(0, 1) * 6))).tanh / (1 + (drive.clip(0, 1) * 0.5));
        sig = ~morphFilter.(sig, cutoff.clip(40, 16000), res, fmode);
        // onset marker so the note reads against the drone (as classic)
        sig = sig + (HPF.ar(src, 2000) * EnvGen.kr(Env.perc(0.001, 0.06)) * 0.3);

        sig = ~safe.(LeakDC.ar(sig));
        Out.ar(out, Pan2.ar(sig * env * amp.clip(0, 0.8), pan.clip(-1, 1)));
    }).add;

    // ── \bassStutterRich ──────────────────────────────────────────────────
    // Same gesture (low / falling / expanding blob), bigger engine:
    // band-limited square + saw stack, MoogLadder (wobble depth from lfoDepth
    // as before), and a SmoothDecimator whose rate follows stutterRate so a
    // faster chop also gets grittier. `crush` (extra, 0..1) scales the crush.
    // v4.11.4: `harm` (extra, 0..1, default 0.6) adds the bass's own upper
    // harmonics (tanh-saturated copy, band-passed 90 Hz-1.2 kHz, chopped with
    // the stutter). Small speakers cannot reproduce a 45 Hz fundamental, but
    // the ear infers it from its harmonics (the "missing fundamental"), so
    // the bass reads as low AND audible; the real sub stays underneath for
    // speakers that can play it. harm 0 = the v4.10 sound.
    SynthDef(\bassStutterRich, {
        |out = 0, freq = 45, amp = 0.25, pan = 0, gate = 1,
         stutterRate = 12, lfoDepth = 300, crush = 0.5, harm = 0.6|

        var sig, env, lfo, trig, delayTime, freqL, ampL, panL, rateL, depthL, cutL, upper;

        freqL  = freq.lag(0.08).clip(25, 120);
        ampL   = amp.lag(0.10).clip(0, 0.8);
        panL   = pan.lag(0.10).clip(-1, 1);
        rateL  = stutterRate.lag(0.15).clip(1, 48);
        depthL = lfoDepth.lag(0.20).clip(1, 1200);

        sig = (~bandSub.(freqL, 0) * 0.8) + (~bandSub.(freqL * 0.5, 1) * 0.7);

        lfo  = SinOsc.kr(2).range(1, depthL);
        cutL = (freqL * 2 + lfo).clip(35, 4000);
        sig  = if (~has.(\MoogLadder)) {
            ~ug.(\MoogLadder).ar(sig, cutL, 0.35)
        } {
            RLPF.ar(sig, cutL, 0.25)
        };

        trig      = Impulse.kr(rateL);
        delayTime = Latch.kr(Sweep.kr(trig), trig) % (1 / rateL.max(0.1));
        sig       = DelayC.ar(sig, 0.5, delayTime);

        // upper harmonics of the chopped bass (missing-fundamental layer)
        upper = LPF.ar(HPF.ar((sig * 8).tanh, 90), 1200);
        sig   = sig + (upper * harm.lag(0.2).clip(0, 1) * 0.45);

        if (~has.(\SmoothDecimator)) {
            sig = ~ug.(\SmoothDecimator).ar(sig,
                rate: rateL.linexp(1, 48, 22000, 3500) * (1 - (crush * 0.6)).max(0.2),
                smoothing: 0.55);
        };

        env = EnvGen.kr(Env.asr(0.01, 1, 0.15), gate, doneAction: 2);
        sig = Limiter.ar(~safe.(sig * env * ampL * 1.5), 0.9);
        Out.ar(out, Pan2.ar(sig, panL));
    }).add;

    // ── \blobStrikeRich ───────────────────────────────────────────────────
    // v4.11.2: a quiet orchestral swell for births (which cluster around
    // collisions). History: v4.10 was an impulse into an inharmonic Klank +
    // MembraneCircle (metallic startle); the first v4.11 drove Klank and the
    // membrane with 0.3 s of continuous noise, which OVERLOADS them -- Klank /
    // Ringz gain grows with ring time, so sustained input into a 2 s ring is
    // +40-60 dB (the bench measured +61 dBFS peaks). Now:
    //   * pitch folded into 60-500 Hz, partials 1-4 only
    //   * body: a low sine "thud" an octave down with a slight pitch fall
    //     (timpani-like), no resonator at all
    //   * harmonic ring: Klank excited by ONE quiet impulse (the safe use)
    //   * string swell: two detuned DWGBowed (saws through a low-pass without
    //     the plugin) crescendoing over the onset
    //   * a 0.15-0.5 s sine onset on everything, so the impulse never clicks
    //   * low-pass 1.2-2.5 kHz, a Limiter at 0.5 as a hard ceiling, and
    //     ~strikeGain (default 0.3) on the output
    // Same control contract: freq amp decay blobSize position elongation.
    ~strikeGain = ~strikeGain ? 0.3;
    SynthDef(\blobStrikeRich, {
        |freq = 440, amp = 0.4, decay = 2.5, blobSize = 0.5, position = 0.5,
         elongation = 0.0|

        var f, sz, el, onset, rel, env, thud, harm, bow, swellEnv, gateEnv, sig;
        f     = freq.clip(60, 2000);
        f     = f / (2 ** ((f / 500).log2.ceil.max(0)));      // fold into 60..500 Hz
        sz    = blobSize.clip(0, 1);
        el    = elongation.clip(0, 1);
        onset = 0.15 + (0.35 * sz);
        rel   = (decay * (0.5 + (0.7 * sz)) + 0.4).clip(0.6, 5);
        env   = EnvGen.kr(Env([0, 1, 0], [onset, rel], [\sin, -2.5]), doneAction: 2);

        // timpani-ish body: plain sine an octave down, pitch falls 6 % into place
        thud  = SinOsc.ar(f * 0.5 * EnvGen.kr(Env([1.06, 1.0], [onset + 0.2], -3)))
                * EnvGen.kr(Env.perc(onset * 0.5, rel * 0.5, 1, -4)) * 0.5;

        // harmonic ring: one quiet impulse (never a sustained excitation)
        harm  = Klank.ar(`[[1, 2, 3, 4], [1, 0.45, 0.22, 0.1], [1.5, 1.0, 0.7, 0.5]],
                         Impulse.ar(0) * 0.2, f);

        // string-section swell
        gateEnv  = EnvGen.kr(Env([1, 1, 0], [onset + (rel * 0.35), 0.0]));
        swellEnv = EnvGen.kr(Env([0, 1, 0.5, 0], [onset * 1.4, rel * 0.4, rel * 0.6], [\sin, -2, -3]));
        bow   = if (~has.(\DWGBowed)) {
            (~ug.(\DWGBowed).ar(f, velb: 0.3, force: 0.5 + (0.25 * el), gate: gateEnv,
                                pos: 0.13, release: rel * 0.6, c1: 0.25, c3: 6 + (10 * el))
             + ~ug.(\DWGBowed).ar(f * 1.004, velb: 0.28, force: 0.45, gate: gateEnv,
                                  pos: 0.11, release: rel * 0.6, c1: 0.25, c3: 6 + (10 * el))) * 0.8
        } {
            LPF.ar(Saw.ar(f * [1, 1.004, 0.5]).sum * 0.33, (f * (2 + (2 * el))).clip(150, 3000))
        };
        bow   = bow * swellEnv;

        sig = (thud * 0.6) + (harm * 0.3) + (bow * 0.5);
        sig = LPF.ar(sig, (1500 + (1000 * el)).clip(1200, 2500));
        sig = Limiter.ar(~safe.(LeakDC.ar(sig)), 0.5, 0.01);      // hard ceiling
        Out.ar(0, Pan2.ar(sig * env * amp * ~strikeGain, position * 2 - 1));
    }).add;

    // ── \blobAngleRich ────────────────────────────────────────────────────
    // The auxiliary pad doubled on every blob voice (at 1.5x the voice
    // pitch, orientation -> pan). Classic: two sines. Rich: Xenakis dynamic
    // stochastic synthesis (Gendy1) wandering in a narrow band around the
    // pitch, so the harmonic doubling breathes instead of sitting still; the
    // walk's step size follows `orientation` motion through the lag.
    SynthDef(\blobAngleRich, {
        |freq = 220, amp = 0.15, gate = 1, orientation = 0.0, attack = 2.5, release = 5.0|

        var sig, env, pan, f, body, grain;
        pan = orientation.linlin(-1.5708, 1.5708, -1, 1).lag(0.4);
        f   = freq.lag(0.3).clip(40, 4000);
        env = EnvGen.kr(Env.asr(attack, 1, release, curve: \sin), gate, doneAction: 2);
        grain = Gendy1.ar(ampdist: 1, durdist: 3, adparam: 0.4, ddparam: 0.2,
                          minfreq: f * 0.995, maxfreq: f * 1.005,
                          ampscale: 0.08, durscale: 0.05, initCPs: 12, knum: 12);
        body  = SinOsc.ar([f, f * 1.007]).sum * 0.5;
        sig   = LPF.ar((grain * 0.5) + (body * 0.6), (f * 3).clip(200, 9000));
        Out.ar(0, Pan2.ar(LeakDC.ar(sig) * env * amp, pan));
    }).add;

    // ── \preSoundRich ─────────────────────────────────────────────────────
    // Imagination-phase pad: three of these (one per triad tone) are held by
    // synth_palette.PreSoundLayer at a gain equal to the harmony posterior
    // mass on NON-current candidates. VarLag on amp and freq so the coming
    // chord fades up and re-pitches without steps.
    SynthDef(\preSoundRich, {
        |out = 0, freq = 220, amp = 0.0, gate = 1, pan = 0|
        var env, sig, ampL, freqL;
        env   = EnvGen.kr(Env.asr(2.0, 1.0, 3.0), gate, doneAction: 2);
        ampL  = VarLag.kr(amp, 1.5, warp: \linear);
        freqL = VarLag.kr(freq, 1.2, warp: \exp);
        sig = Mix.new([
            SinOsc.ar(freqL),
            SinOsc.ar(freqL * 1.005 * SinOsc.kr(0.07).range(0.998, 1.002), 0, 0.5),
            SinOsc.ar(freqL * 2.01, 0, 0.12)
        ]) * 0.3;
        sig = sig + (~airResonator.(freqL, 0.95) * 0.08);
        sig = LPF.ar(sig, (freqL * 3).clip(80, 12000));
        Out.ar(out, Pan2.ar(sig * env * ampL, pan.clip(-1, 1)));
    }).add;

    // ── \lavaMaster ───────────────────────────────────────────────────────
    // Tail insert on bus 0. Controls (all lagged; Python sends every frame):
    //   wet       0..1   reverb wet fraction            <- phrase position
    //   t60       s      reverb time                    <- ambiguity
    //   size      0.5..5 room size                      <- blob count (fewer = bigger)
    //   damp      0..1   HF damping of the tail         <- 1 - tension
    //   modDepth  0..1   internal chorus of the tail    <- 1 - Kuramoto order
    //   holeMix   0..1   Greyhole (metallic drift) send <- tension
    //   holeFb    0..1   Greyhole feedback              <- surprisal
    //   highcut   Hz     tail LPF                       <- register centre
    //   drive     0..1   pre-reverb saturation          <- tension
    //   t_bloom   trig   boundary bloom (feedback swell)<- grouping boundary
//                    (a trigger control: /n_set t_bloom 1 fires it once)
    //   bloomAmt  0..1   bloom depth                    <- boundary strength
    //   bypass    0/1    1 = pass the dry mix through unchanged
    SynthDef(\lavaMaster, {
        |wet = 0.35, t60 = 4.0, size = 2.0, damp = 0.5, modDepth = 0.3,
         holeMix = 0.1, holeFb = 0.4, highcut = 6000, drive = 0.0,
         t_bloom = 0, bloomAmt = 0.0, bypass = 0|

        var dry, src, verb, hole, bloomEnv, fb, sig, wetL, holeL, drvL;

        // v4.11.3: scrub the input so one bad voice cannot poison the
        // reverb / Greyhole feedback (and through them the whole output)
        dry  = ~safe.(In.ar(0, 2), 2.0);
        wetL = wet.lag(1.5).clip(0, 0.9);
        holeL = holeMix.lag(2.0).clip(0, 0.6);
        drvL = drive.lag(1.0).clip(0, 1);

        src = XFade2.ar(dry, (dry * (1 + (drvL * 3))).tanh, (drvL * 2) - 1);

        // boundary bloom: a short swell of feedback and wet level
        bloomEnv = EnvGen.kr(Env([0, 1, 0], [0.4, 3.5], [2, -3]), t_bloom) * bloomAmt.clip(0, 1);

        verb = if (~has.(\JPverb)) {
            ~ug.(\JPverb).ar(src,
                t60: (t60.lag(3.0) * (1 + (bloomEnv * 1.5))).clip(0.3, 40),
                damp: damp.lag(2.0).clip(0, 1),
                size: size.lag(4.0).clip(0.5, 5),
                earlyDiff: 0.707,
                modDepth: modDepth.lag(2.0).clip(0, 1),
                modFreq: 2.0,
                low: 1.0, mid: 1.0, high: 0.6,
                lowcut: 120, highcut: highcut.lag(2.0).clip(1000, 12000))
        } {
            FreeVerb2.ar(src[0], src[1], 1.0, (t60 / 12).clip(0.2, 0.98), damp.clip(0, 1))
        };

        fb   = (holeFb.lag(2.0).clip(0, 0.95) + (bloomEnv * 0.5)).clip(0, 0.97);
        hole = if (~has.(\Greyhole)) {
            ~ug.(\Greyhole).ar(src, delayTime: 0.25, damp: 0.3, size: 2.5, diff: 0.707,
                               feedback: fb, modDepth: modDepth.lag(2.0).clip(0, 1) * 0.6,
                               modFreq: 0.4)
        } {
            CombC.ar(src, 0.5, 0.25, fb * 6)
        };

        sig = (dry * (1 - wetL)) + (verb * (wetL + (bloomEnv * 0.25))) + (hole * (holeL + (bloomEnv * 0.2)));
        sig = Limiter.ar(LeakDC.ar(sig), 0.92, 0.01);
        sig = Select.ar(bypass.clip(0, 1), [sig, dry]);
        ReplaceOut.ar(0, sig);
    }).add;

    // ── \lavaMasterVST (optional) ─────────────────────────────────────────
    // Same insert, VST3 as the space. p0..p7 are the plugin's first eight
    // parameters (0..1), so Python can map ScoreState onto them with /n_set.
    // Open the plugin from the language: see ~openMasterVST below.
    if (~has.(\VSTPlugin)) {
        SynthDef(\lavaMasterVST, {
            |wet = 0.5, bypass = 0, p0 = 0.5, p1 = 0.5, p2 = 0.5, p3 = 0.5,
             p4 = 0.5, p5 = 0.5, p6 = 0.5, p7 = 0.5|
            var dry, fx, sig;
            dry = In.ar(0, 2);
            fx  = ~ug.(\VSTPlugin).ar(dry, numOut: 2,
                     params: [0, p0.lag(1), 1, p1.lag(1), 2, p2.lag(1), 3, p3.lag(1),
                              4, p4.lag(1), 5, p5.lag(1), 6, p6.lag(1), 7, p7.lag(1)]);
            sig = XFade2.ar(dry, fx, (wet.lag(1.5).clip(0, 1) * 2) - 1);
            sig = Limiter.ar(sig, 0.92, 0.01);
            sig = Select.ar(bypass.clip(0, 1), [sig, dry]);
            ReplaceOut.ar(0, sig);
        }).add;
    };

    s.sync;
    s.sendMsg("/g_new", 1, 0, 0);     // restore default group if Python removed it

    // Language-side helpers (manual tests / sessions without Python):
    // Python's VoiceManager sends /g_freeAll 0 at startup, which frees any
    // master node spawned here first -- so in a live run synth_palette spawns
    // the master itself (node 900) after VoiceManager has started.
    ~spawnMaster = {
        ~master.free; ~master = Synth.tail(s, \lavaMaster);
        "lavaMaster spawned at the tail of the default group".postln;
        ~master
    };
    ~openMasterVST = { |name = "ValhallaSupermassive"|
        if (~has.(\VSTPlugin)) {
            ~masterVST.free;
            ~masterVST = Synth.tail(s, \lavaMasterVST);
            ~masterVSTctl = ~ug.(\VSTPluginController).new(~masterVST);
            ~masterVSTctl.open(name, editor: true, action: { |vst|
                ("lavaMasterVST: " ++ name ++ " loaded").postln;
                vst.editor;
            });
        } { "VSTPlugin extension not installed".postln };
    };

    ("Audible Lava v4.10: rich palette ready (mpeVoiceRich, mpeHitRich, melodyHitRich, "
     ++ "bassStutterRich, blobStrikeRich, blobAngleRich, preSoundRich, lavaMaster"
     ++ (if (~has.(\VSTPlugin)) { ", lavaMasterVST" } { "" }) ++ ")").postln;
});


// ── Manual tests (evaluate one block at a time after loading) ───────────────
/*

s.sendMsg("/g_new", 1, 0, 0);

// master space first (hear everything below through it); bypass to A/B
~spawnMaster.();
~master.set(\wet, 0.5, \t60, 8, \damp, 0.3, \holeMix, 0.2, \holeFb, 0.6);
~master.set(\t_bloom, 1, \bloomAmt, 0.8);    // a grouping boundary
~master.set(\bypass, 1);  ~master.set(\bypass, 0);

// sustained voice -- the same controls the blob routes drive
(
x = Synth(\mpeVoiceRich, [\freq, 110, \amp, 0.25, \cutoff, 1200, \res, 0.4,
    \wave, 0.5, \bank, 2, \attack, 1.0, \release, 3.0]);
)
x.set(\detune, 0.5);   x.set(\shimmer, 0.7);   x.set(\noise, 0.6);
x.set(\drive, 0.8);    x.set(\fmode, 0.5);     x.set(\sub, 0.7);
x.set(\formant, 0.0);  x.set(\formant, 1.0);   x.set(\chorus, 0.8);
x.set(\gate, 0);

// hits: ornament (plucked) vs paired head (piano) vs classic stack
Synth(\mpeHitRich, [\freq, 330, \amp, 0.4, \dur, 0.3, \vel, 0.9, \model, 1]);
Synth(\mpeHitRich, [\freq, 110, \amp, 0.5, \dur, 0.6, \vel, 0.8, \model, 2]);
Synth(\mpeHitRich, [\freq, 220, \amp, 0.4, \dur, 0.2, \vel, 0.7, \model, 0]);

// melody: legato (bowed) vs staccato (plucked), auto-selected by release
Synth(\melodyHitRich, [\freq, 262, \amp, 0.3, \dur, 1.2, \release, 1.2, \bow, 0.2]);
Synth(\melodyHitRich, [\freq, 392, \amp, 0.3, \dur, 0.3, \release, 0.25]);
Synth(\melodyHitRich, [\freq, 330, \amp, 0.3, \dur, 1.5, \release, 1.2, \bow, 0.9]);

// bass gesture
b = Synth(\bassStutterRich, [\freq, 45, \amp, 0.3, \stutterRate, 12, \lfoDepth, 300]);
b.set(\stutterRate, 26, \lfoDepth, 700, \crush, 0.9);
b.set(\gate, 0);

// births (v4.11: a swell, not a clang) and the pad
Synth(\blobStrikeRich, [\freq, 300, \amp, 0.5, \decay, 2.5, \blobSize, 0.8, \elongation, 0.7]);
Synth(\blobStrikeRich, [\freq, 180, \amp, 0.5, \decay, 1.5, \blobSize, 0.2, \elongation, 0.0]);
a = Synth(\blobAngleRich, [\freq, 330, \amp, 0.15, \orientation, 0.9]);
a.set(\gate, 0);

// pre-sound: the predicted triad fading up as posterior mass shifts
(
p = [196, 246.9, 293.7].collect { |f| Synth(\preSoundRich, [\freq, f, \amp, 0.0]) };
)
p.do(_.set(\amp, 0.06));   p.do(_.set(\amp, 0.0));   p.do(_.set(\gate, 0));

// Pbind audition of the rich hit family (language clock -- for listening only;
// the live system schedules hits from Python's elastic clock, never a TempoClock)
(
~audition = Pbind(
    \instrument, \mpeHitRich,
    \degree, Pseq([0, 2, 4, 7, 9, 7, 4, 2], inf),
    \scale, Scale.majorPentatonic, \octave, Pwhite(3, 5, inf).stutter(8),
    \dur, Prand([0.25, 0.5, 0.75], inf),
    \amp, Pwhite(0.2, 0.4, inf), \vel, Pwhite(0.4, 1.0, inf),
    \model, Pseq([1, 1, 2, 1, 0], inf),
    \drive, Pwhite(0.0, 0.5, inf), \noise, Pwhite(0.0, 0.3, inf),
    \pan, Pwhite(-0.6, 0.6, inf)
).play;
)
~audition.stop;

// VST master instead of \lavaMaster (needs the VSTPlugin extension + a plugin)
~useVST = true;
~openMasterVST.("ValhallaSupermassive");
~masterVST.set(\wet, 0.6, \p0, 0.8, \p1, 0.3);

*/
