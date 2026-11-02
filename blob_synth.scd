// blob_synth.scd -- Audible Lava / DSO v4.0 (MPE Edition)
//
// v4.9.2 (Oct 2026) -- THE WAVETABLES WERE NEVER FILLED.
//   The fill helpers called buf.sine1Msg(amps). In SuperCollider, *Msg
//   methods only BUILD the OSC message and return it; they never send it.
//   buf.sine1(amps) is the call that fills the buffer. So all 16 wavetable
//   buffers have been zeros since this file was written, \mpeVoice (and
//   \mpeHit, \melodyHit -- anything on VOsc) has been silent, and what was
//   audible as "the drone" was \blobAngle (a sine pad at 1.5x the voice
//   pitch), \bassStutter and the \blobStrike births. Fixed: every sine1Msg
//   is now sine1, and the fill is verified by reading buffer 0 back.
//
//   Second, related: a dummy Buffer.alloc(s, 1024) was taking buffer id 0,
//   so the 16 wavetables landed on ids 1..16 while every SynthDef defaults
//   bufBase = 0 and nothing on the Python side sends bufBase. VOsc then
//   straddled the 1024-sample dummy and the first 2048-sample table, which
//   is a size mismatch VOsc answers with silence. The wavetables are now
//   allocated explicitly at ids 0..15 (allocConsecutive's bufnum: arg), so
//   bufBase = 0 is correct and re-evaluating this file without a reboot
//   re-fills the same ids rather than allocating a fresh range.
//
// SynthDefs:
//   \mpeVoice    -- v4.0 — wavetable osc, SVF filter, ADSR, the polysynth
//                          voice that the blob CV features and the player
//                          keyboard share. One instance per active voice
//                          slot. Receives /n_set updates every frame.
//   \blobAngle   -- v3.x — orientation-panned sine pad (still spawned by
//                          BLOB_ONLY/HYBRID voices for harmonic doubling)
//   \blobStrike  -- v3.x — Klank transient on blob birth
//   \bassStutter -- low sub-bass stutter used when an existing blob is
//                   falling lower in the lamp while expanding
//   \blobVoice   -- v3.x — kept for back-compat / fallback
//
// The wavetable buffers are loaded into a contiguous range of buffer ids,
// 4 banks × 4 wavetables = 16 buffers total (~bufBase .. ~bufBase+15).
// VOsc reads the 'wave' index into this range, so e.g. wave=0 bank=0 is
// buffer ~bufBase (saw), wave=1 bank=2 is ~bufBase + 9 (formant 'eh').
//
// If the server reports a sample-rate error on boot: open Audio MIDI Setup,
// set the output device (e.g. Pebble V3) to 44100 or 48000 Hz, and make the
// input device match -- or keep numInputBusChannels = 0 as below so the input
// device is never opened.

//s.freeAll;
//s.reboot;
//s.options.outDevice = "Pebble V3";
s.options.sampleRate         = 48000;
s.options.numInputBusChannels = 0;
s.boot;

s.waitForBoot({

    // ── Wavetable buffers ─────────────────────────────────────────────────
    // 4 banks × 4 waves each = 16 contiguous buffers for VOsc, pinned to
    // ids 0..15 so the SynthDefs' bufBase = 0 default is always right.
    ~bufBase   = 0;
    ~wtBuffers = Buffer.allocConsecutive(16, s, 2048, 1, bufnum: ~bufBase);

    // Helper: write a wavetable signal into a buffer (sine-sum form for VOsc).
    // sine1 (not sine1Msg) SENDS the fill; asWavetable: true is the default
    // and is what VOsc requires.
    ~makeSawTable = { |buf, harmonics=24|
        var amps = (1..harmonics).reciprocal;
        buf.sine1(amps);
    };
    ~makeSquareTable = { |buf, harmonics=24|
        var amps = (1..harmonics).collect{ |i|
            ((i + 1) % 2 == 0).if({ 0.0 }, { i.reciprocal })
        };
        buf.sine1(amps);
    };
    ~makeTriTable = { |buf, harmonics=12|
        var amps = (1..harmonics).collect{ |i|
            ((i + 1) % 2 == 0).if({ 0.0 }, { (i * i).reciprocal })
        };
        buf.sine1(amps);
    };
    ~makePulseTable = { |buf|
        var amps = [1, 0.4, 0.7, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2, 0.0, 0.15];
        buf.sine1(amps);
    };

    s.sync;

    // Bank 0 — analogue shapes
    ~makeSawTable.value(~wtBuffers[0]);
    ~makeSquareTable.value(~wtBuffers[1]);
    ~makePulseTable.value(~wtBuffers[2]);
    ~makeTriTable.value(~wtBuffers[3]);

    // Bank 1 — harmonic series sweeps
    [4, 8, 12, 16].do{ |h, i|
        ~wtBuffers[4 + i].sine1((1..h).reciprocal);
    };

    // Bank 2 — formants (vowel-like)
    [
        [1.0, 0.5, 0.7, 0.3, 0.2, 0.1, 0.05],          // ah
        [1.0, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1],           // eh
        [1.0, 0.9, 0.4, 0.2, 0.1, 0.05, 0.02],         // ih
        [1.0, 0.4, 0.2, 0.5, 0.3, 0.1, 0.05]           // oh
    ].do{ |amps, i|
        ~wtBuffers[8 + i].sine1(amps);
    };

    // Bank 3 — metallic / bell partials
    [
        [1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.1, 0.0],
        [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05],
        [1.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.4, 0.0],
        [1.0, 0.3, 0.5, 0.2, 0.4, 0.15, 0.25, 0.1]
    ].do{ |amps, i|
        ~wtBuffers[12 + i].sine1(amps);
    };

    s.sync;

    // Verify the fill actually landed (this is the check that would have
    // caught the sine1Msg bug): read 8 samples of buffer 0 back.
    ~wtBuffers[0].getn(0, 8, { |d|
        if (d.abs.maxItem < 1e-6) {
            "*** blob_synth.scd: wavetable buffer 0 is EMPTY -- VOsc voices will be silent".postln;
        } {
            ("blob_synth.scd: wavetables filled (buffer 0 head: " ++ d.round(0.001) ++ ")").postln;
        };
    });

    // ── \mpeVoice (v4.1) ──────────────────────────────────────────────────
    // v4.1 adds four expressive dimensions driven by blob motion / change of
    // state. Every new arg defaults to 0.0 and is engineered so that the
    // all-zero case is bit-identical to the v4.0 voice:
    //   detune  — 3-osc unison spread (±4% at 1.0): elongation / speed → beating
    //   shimmer — vibrato + tremolo (~5.2 Hz): turning / acceleration → shimmer
    //   noise   — pitch-tracking band-passed air: dissolving topology → breath
    //   drive   — tanh wavefold: area expansion → harmonic richness
    SynthDef(\mpeVoice, {
        |out      = 0,
         freq     = 220,
         amp      = 0.0,
         pan      = 0,
         gate     = 1,
         attack   = 1.5,
         decay    = 0.4,
         sustain  = 0.7,
         release  = 2.5,
         cutoff   = 1500,
         res      = 0.30,
         fmode    = 0.0,           // 0=LP, 0.5=BP, 1=HP
         wave     = 0,             // 0..3 within bank
         bank     = 0,             // 0..3
         bufBase  = 0,             // base buffer index of the 16-wt bank
         detune   = 0.0,           // 0..1 unison spread / beating
         shimmer  = 0.0,           // 0..1 vibrato + tremolo depth
         noise    = 0.0,           // 0..1 breath / air blend
         drive    = 0.0|           // 0..1 wavefold saturation

        var sig, env, lp, bp, hp, mix;
        var idx, freqL, ampL, cutL, resL, panL, waveL;
        var detL, shimL, noiL, drvL;
        var vib, trem, spread, oscs, folded, air;

        // Smoothing (.lag) on continuous params; these match Python's
        // first-order smoothing nicely so changes never click.
        // Linear VarLag (constant-time glide) rather than exponential .lag —
        // matches the pitchShifter reference; incremental +/-Hz nudges from
        // freq_offset_hz arrive as clean linear ramps.
        freqL   = VarLag.kr(freq, 0.05, warp: \linear);
        ampL    = amp.lag(0.10);
        cutL    = cutoff.lag(0.05).clip(40, 16000);
        resL    = res.lag(0.05).clip(0.05, 1.0);
        panL    = pan.lag(0.05).clip(-1.0, 1.0);
        waveL   = wave.lag(0.05).clip(0, 3.999);
        detL    = detune.lag(0.10).clip(0.0, 1.0);
        shimL   = shimmer.lag(0.10).clip(0.0, 1.0);
        noiL    = noise.lag(0.10).clip(0.0, 1.0);
        drvL    = drive.lag(0.10).clip(0.0, 1.0);

        // Buffer index = bufBase + bank*4 + wave   (continuous)
        idx     = (bufBase + (bank.clip(0, 3) * 4) + waveL);

        env = EnvGen.kr(
            Env.adsr(
                attackTime:  attack.lag(1.0).max(0.005),
                decayTime:   decay.max(0.005),
                sustainLevel: sustain.clip(0.0, 1.0),
                releaseTime: release.lag(1.0).max(0.005),
                curve: \sin),
            gate, doneAction: 2
        );

        // Shimmer: vibrato (pitch) + tremolo (amp). At shimL=0 both collapse
        // to a constant 1.0 (no modulation, no extra audible state).
        vib  = SinOsc.kr(5.2 + (shimL * 1.5)).range(1.0 - (shimL * 0.03),
                                                    1.0 + (shimL * 0.03));
        trem = SinOsc.kr(5.2, 0.5 * pi).range(1.0 - (shimL * 0.35), 1.0);

        // Detuned unison: three VOsc copies spread by detL (±4% at full).
        // At detL=0 the three are identical, so the averaged sum equals the
        // single v4.0 oscillator exactly.
        spread = detL * 0.04;
        oscs = [
            VOsc.ar(idx, freqL * vib,                  mul: 1.0),
            VOsc.ar(idx, freqL * vib * (1.0 - spread), mul: 1.0),
            VOsc.ar(idx, freqL * vib * (1.0 + spread), mul: 1.0)
        ];
        sig = Mix.new(oscs) * (1.0 / 3.0);

        // Drive: tanh wavefold, crossfaded in by drvL (drvL=0 → fully dry).
        folded = (sig * (1.0 + (drvL * 4.0))).tanh;
        sig    = XFade2.ar(sig, folded, (drvL * 2.0) - 1.0);

        // SVF crossfade: LP <-> BP <-> HP via fmode 0..1
        lp  = RLPF.ar(sig, cutL, resL.linlin(0.05, 1.0, 1.0, 0.05));
        bp  = BPF.ar( sig, cutL, resL.linlin(0.05, 1.0, 1.0, 0.05));
        hp  = RHPF.ar(sig, cutL, resL.linlin(0.05, 1.0, 1.0, 0.05));
        mix = SelectX.ar( fmode.lag(0.05).clip(0.0, 1.0) * 2.0, [lp, bp, hp] );

        // Breath / air: band-passed pink noise tracking the fundamental,
        // blended in by noiL (noiL=0 → silent, no signal change).
        air = BPF.ar(PinkNoise.ar(1.0), freqL.clip(60, 8000), 0.5) * 6.0;
        mix = mix + (air * noiL * 0.5);

        Out.ar(out, Pan2.ar(mix * env * ampL * trem, panL));
    }).add;

    // ── \blobVoice (v3.x; kept for back-compat) ───────────────────────────
    SynthDef(\blobVoice, {
        |out=0, freq=220, amp=0, pan=0, gate=1, cutoff=1200, res=0.3,
         elongation=0.0, attack=2.0, rel=3.5|
        var sig, env, filt, res_mod;
        res_mod = res + (elongation.lag(0.5) * 0.5);
        env = EnvGen.kr(
            Env.asr(attack.lag(1.0), 1, rel.lag(1.0), curve: \sin),
            gate, doneAction: 2
        );
        sig  = Saw.ar(freq.lag(0.5));
        filt = RLPF.ar(sig, cutoff.lag(0.4).clip(80, 12000),
                       res_mod.clip(0.05, 0.95));
        Out.ar(out, Pan2.ar(filt * env * amp.lag(0.3), pan.lag(0.2)));
    }).add;

    // ── \blobAngle (v3.x; orientation-panned harmonic pad) ────────────────
    SynthDef(\blobAngle, {
        |freq=220, amp=0.15, gate=1, orientation=0.0,
         attack=2.5, release=5.0|
        var sig, env, pan;
        pan = orientation.linlin(-1.5708, 1.5708, -1.0, 1.0).lag(0.4);
        env = EnvGen.kr(
            Env.asr(attack, 1, release, curve: \sin),
            gate, doneAction: 2
        );
        sig = SinOsc.ar([freq, freq * 1.007]);
        Out.ar(0, Pan2.ar(LeakDC.ar(sig.sum) * env * amp, pan));
    }).add;

    // ── \blobStrike (v3.x; Klank transient on birth) ──────────────────────
    SynthDef(\blobStrike, {
        |freq=440, amp=0.4, decay=2.5, blobSize=0.5, position=0.5,
         elongation=0.0|
        var sig, env, freqMod;
        freqMod = freq * (1.0 + elongation.clip(0, 1) * 0.12);
        env = EnvGen.kr(Env.perc(0.001, decay * blobSize + 0.1),
                        doneAction: 2);
        sig = Klank.ar(
            `[[1, 2.1, 3.3, 4.7, 6.2], nil, [1.5, 0.8, 0.6, 0.4, 0.2]],
            Impulse.ar(0) + WhiteNoise.ar(0.01),
            freqMod
        );
        Out.ar(0, Pan2.ar(sig * env * amp, position * 2 - 1));
    }).add;

    // ── \bassStutter (gesture transform for existing low/falling/growing blobs) ──
    // Spawned by voice_manager.py as a replacement main node on the same voice
    // slot.  Continuous /n_set updates drive freq, amp, pan, stutterRate, and
    // lfoDepth from the blob's y position, downward velocity, and area growth.
    SynthDef(\bassStutter, {
        |out = 0,
         freq = 45,
         amp = 0.25,
         pan = 0,
         gate = 1,
         stutterRate = 12,
         lfoDepth = 300|

        var sig, env, lfo, trig, delayTime;
        var freqL, ampL, panL, rateL, depthL;

        freqL  = freq.lag(0.08).clip(25, 120);
        ampL   = amp.lag(0.10).clip(0.0, 0.8);
        panL   = pan.lag(0.10).clip(-1.0, 1.0);
        rateL  = stutterRate.lag(0.15).clip(1.0, 48.0);
        depthL = lfoDepth.lag(0.20).clip(1.0, 1200.0);

        // Low oscillator stack: saw edge plus sub pulse.
        sig = Saw.ar(freqL) + Pulse.ar(freqL * 0.5, 0.5);

        // Slow wobbling low-pass movement, stronger when the blob falls faster.
        lfo = SinOsc.kr(2).range(1, depthL);
        sig = RLPF.ar(sig, (freqL + lfo).clip(35, 4000), 0.2);

        // Stuttered delay-time latch.  Sweep resets each impulse, so the delay
        // pattern tightens as stutterRate rises with area expansion.
        trig = Impulse.kr(rateL);
        delayTime = Latch.kr(Sweep.kr(trig), trig) % (1 / rateL.max(0.1));
        sig = DelayC.ar(sig, 0.5, delayTime);

        env = EnvGen.kr(Env.asr(0.01, 1, 0.15), gate, doneAction: 2);
        sig = Limiter.ar(sig * env * ampL * 1.5, 0.9);

        Out.ar(out, Pan2.ar(sig, panL));
    }).add;

    s.sync;

    // Python's VoiceManager sends /g_freeAll 0 at startup, which also removes
    // the default group (node 1). Nothing Python does needs it -- it spawns
    // into group 0 -- but language-side Synth(...) tests do. voice_manager
    // v4.9.2 recreates it; this line covers the case where this file is
    // evaluated after Python has already started.
    s.sendMsg("/g_new", 1, 0, 0);

    "Audible Lava v4.9.2: SynthDefs ready (mpeVoice, blobVoice, blobAngle, blobStrike, bassStutter)".postln;
    ("Wavetable buffers: " ++ ~wtBuffers[0].bufnum ++ " .. " ++ ~wtBuffers[15].bufnum).postln;
});


// ── Manual tests ─────────────────────────────────────────────────────────────
//
// These snippets are designed to be evaluated INTERACTIVELY, one block at a
// time, AFTER the SynthDefs have loaded. Place the cursor inside a parens
// block (or on a single line) and press shift+return.
//
// They are commented out so the file as a whole evaluates cleanly without
// hitting "expecting end of file" errors when SC parses multiple top-level
// statements outside of a paren group.

/*

// If Python has been started since the server booted, restore the default
// group first (Python's /g_freeAll 0 removes it):
s.sendMsg("/g_new", 1, 0, 0);

// Wavetable sanity: this must sound. If it does not, the fill failed.
{ VOsc.ar(0, 220, mul: 0.3) ! 2 }.play;    // Cmd+. to stop

// Spawn an mpeVoice with bank 0 / wave 0 (saw):
(
x = Synth(\mpeVoice, [
    \freq, 220, \amp, 0.25, \cutoff, 1500, \res, 0.4,
    \attack, 1.0, \release, 2.5, \wave, 0, \bank, 0,
    \fmode, 0.0, \bufBase, ~wtBuffers[0].bufnum,
]);
)

// Sweep — evaluate any of these lines individually (cursor on line, shift+return):
x.set(\cutoff, 4000);
x.set(\fmode, 1.0);          // -> HP
x.set(\bank, 2, \wave, 1);   // -> formant 'eh'

// v4.1 expressive params (each 0..1) — these are what blob motion drives:
x.set(\detune, 0.6);         // unison spread / beating (elongation, speed)
x.set(\shimmer, 0.8);        // vibrato + tremolo (turning, acceleration)
x.set(\noise, 0.5);          // breath / air (dissolving topology)
x.set(\drive, 0.7);          // wavefold richness (area expansion)
x.set(\detune, 0, \shimmer, 0, \noise, 0, \drive, 0);  // back to clean v4.0 tone
x.set(\gate, 0);


// bassStutter manual test:
(
b = Synth(\bassStutter, [\freq, 45, \amp, 0.25, \pan, 0,
                           \stutterRate, 12, \lfoDepth, 300]);
)
b.set(\freq, 38, \stutterRate, 22, \lfoDepth, 600, \amp, 0.35);
b.set(\gate, 0);

// Two voices: bank 0 saw + bank 2 formant
(
y = Synth(\mpeVoice, [\freq, 110, \amp, 0.20, \cutoff,  600,
                       \wave, 0, \bank, 0,
                       \bufBase, ~wtBuffers[0].bufnum]);
z = Synth(\mpeVoice, [\freq, 220, \amp, 0.20, \cutoff, 2000,
                       \wave, 0, \bank, 2,
                       \bufBase, ~wtBuffers[0].bufnum]);
)
[y, z].do(_.set(\gate, 0));

*/
