
//s.freeAll;
//s.reboot;
(
// 1. SynthDef can be evaluated safely at the top level
SynthDef(\pitchShifter, { |freq = 440, amp = 0.2|
    var sig;
    var smoothFreq = VarLag.kr(freq, 0.05, warp: \linear);

    sig = SinOsc.ar(smoothFreq);
    sig = sig * amp ! 2;
    OffsetOut.ar(0, sig);
}).add;

// 2. Wrap server instantiation and GUI setup inside a fork
fork {
    var win, width = 300, height = 150;

    // Wait for the server to successfully compile and register the SynthDef
    s.sync;

    // Defer the GUI creation to the AppClock (required when inside a fork)
    defer {
        // Initialize variables and start the synth instance
        ~currentFreq = 440;
        ~mySynth = Synth(\pitchShifter, [\freq, ~currentFreq]);

        // Build and launch the control window
        win = Window(" left arrow down 5Hz/ right arrow up 5Hz", Rect(200, 200, width, height)).front;

        win.view.keyDownAction_({ |view, char, modifiers, unicode, keycode|
            case
                { char == $a } {
                    ~currentFreq = ~currentFreq + 5;
                    ~mySynth.set(\freq, ~currentFreq);
                    ("Frequency increased to: " + ~currentFreq + "Hz").postln;
                }

                { char == $d } {
                    ~currentFreq = (~currentFreq - 5).max(5);
                    ~mySynth.set(\freq, ~currentFreq);
                    ("Frequency decreased to: " + ~currentFreq + "Hz").postln;
                };
        });

        win.onClose_({
            ~mySynth.free;
            "Synth freed.".postln;
        });
    };
};
)



//    choal soprano

// 1. Boot the audio server first if it isn't running
s.boot;

// 2. Define the Soprano Choir SynthDef
(
SynthDef(\sopranoChoir, { arg freq = 523.25, // Default to C5 (High Soprano register)
    amp = 0.2, gate = 1, vowSelect = 0,      // 0 = "Ah", 1 = "Oh"
    detune = 0.012;

    var vibrato, source, pulseSrc, noiseSrc, formants, env, signal;
    var f1, f2, f3, a1, a2, a3, bw1, bw2, bw3;

    // --- 1. VIBRATO & JITTER ---
    vibrato = SinOsc.kr(LFNoise1.kr(2).exprange(5.5, 6.5)) * (freq * 0.025);
    vibrato = vibrato + LFNoise1.kr(10, freq * 0.003);

    // --- 2. VOCAL SOURCE ---
    pulseSrc = Mix.new([
        Saw.ar(freq + vibrato),
        Pulse.ar((freq * 1.002) + vibrato, 0.35) * 0.5,
        Saw.ar(freq * (1 + detune) + vibrato) * 0.3,
        Saw.ar(freq * (1 - detune) + vibrato) * 0.3
    ]);

    noiseSrc = HPF.ar(WhiteNoise.ar(0.1), 2000) * LFNoise1.kr(5).range(0.5, 1);
    source = (pulseSrc * 0.85) + (noiseSrc * 0.15);

    // --- 3. FORMANT DATA ---
    f1 = Select.kr(vowSelect, [800, 600]);
    f2 = Select.kr(vowSelect, [1150, 950]);
    f3 = Select.kr(vowSelect, [2900, 2800]);

    // Simplified dB to Amplitude conversion directly using .dbamp
    // to prevent ControlSpec mapping issues inside the SynthDef
    a1 = Select.kr(vowSelect, [0, 0]).dbamp;
    a2 = Select.kr(vowSelect, [-6, -11]).dbamp;
    a3 = Select.kr(vowSelect, [-32, -22]).dbamp;

    bw1 = Select.kr(vowSelect, [80, 80]);
    bw2 = Select.kr(vowSelect, [90, 90]);
    bw3 = Select.kr(vowSelect, [120, 120]);

    // --- 4. FORMANT FILTER BANK ---
    formants = Mix.new([
        BBandPass.ar(source, f1, bw1 / f1) * a1,
        BBandPass.ar(source, f2, bw2 / f2) * a2,
        BBandPass.ar(source, f3, bw3 / f3) * a3
    ]);

    formants = BHiShelf.ar(formants, 3200, 1, 4);

    // --- 5. ENVELOPE ---
    env = EnvGen.kr(Env.asr(attackTime: 0.6, sustainLevel: 1.0, releaseTime: 1.2), gate, doneAction: 2);

    // --- 6. STEREO FIELD & REVERB ---
    // FIX: Splay spreads an array of mono signals.
    // We pass a multi-channel detuned array instead of a manually delayed pair.
    signal = Splay.ar([formants, DelayC.ar(formants, 0.02, 0.015)]);

    signal = FreeVerb.ar(signal, mix: 0.45, room: 0.85, damp: 0.3);

    // Multiply by env and amp here to ensure it applies cleanly to both stereo channels
    Out.ar(0, signal * env * amp);
}).add;
)

// 3. Play a sustained High Soprano C (C5) singing "Ah"
x = Synth(\sopranoChoir, [\freq, 523.25, \vowSelect, 0, \amp, 0.25]);

// 4. Change the vowel dynamically to "Oh" while playing
x.set(\vowSelect, 1);

// 5. Change it back to "Ah"
x.set(\vowSelect, 0);

// 6. Release the note smoothly
x.set(\gate, 0);


//
//
//   generative plugin gendy3
//
// 1. Play the Gendy3 example directly
(
x = {
    var sig, env;

    // Control parameters dynamically with your mouse
    // X-axis: Min frequency / overall pitch register
    // Y-axis: Max frequency / bandwidth of the variation
    var minFreq = MouseX.kr(20, 400, \exponential);
    var maxFreq = MouseY.kr(100, 3000, \exponential);

    sig = Gendy3.ar(
        ampdist: 1,       // Amplitude distribution (1 = uniform)
        durdist: 3,       // Duration distribution (3 = cauchy, very volatile)
        adparam: 1.0,     // Distribution parameter for amplitude
        ddparam: 1.0,     // Distribution parameter for duration
        minfreq: minFreq, // Lower bound of the pitch
        maxfreq: maxFreq, // Upper bound of the pitch
        ampscale: 0.5,    // Scale factor for amplitude change
        durscale: 0.5,    // Scale factor for duration change
        initCPs: 12,      // Number of initial control points (higher = more complex)
        mul: 0.2          // Keep volume safe; Gendy can get loud!
    );

    // Duplicate to stereo with a slight delay for space
    [sig, DelayC.ar(sig, 0.02, 0.015)] * 0.5;
}.play;
)

// 2. Stop the sound
x.free;


//
//
//
//    JPVerb: lush ambient sounds
//
// 1. Play the JPVerb ambient space example
(
x = {
    var source, reverb;

    // --- 1. THE SOURCE: Short, random metallic plucks ---
    source = Dust.ar(1.5); // Random triggers roughly 1.5 times per second
    source = Decay2.ar(source, 0.001, 0.05); // Sharp attack, fast decay envelope
    source = source * SinOsc.ar(LFNoise1.kr(0.5).exprange(200, 800)); // Modulate the pitch of the pluck
    source = Pan2.ar(source, LFNoise1.kr(1)); // Randomly pan the dry clicks left/right

    // --- 2. THE REVERB: JPVerb ---
    reverb = JPverb.ar(
        source,
        t60: 8.0,          // Reverb time in seconds (8 seconds = massive trail!)
        damp: 0.8,         // High frequency damping (0 = crisp, 1 = dark/warm)
        size: 2.0,         // Room size scale factor (larger = more spacious)
        earlyDiff: 0.707,  // Shape of early reflections
        modDepth: 1.0,     // Internal chorus modulation depth
        modFreq: 4.0,      // Internal chorus modulation speed in Hz
        low: 1.0,          // Low-frequency multiplier
        mid: 1.0,          // Mid-frequency multiplier
        high: 0.5,         // High-frequency multiplier (tames harsh high end in the tail)
        lowcut: 100,       // HPF to cut muddy low-end build up
        highcut: 6000      // LPF to keep the tail warm and distant
    );

    // Mix dry source and wet reverb together
    // 0.2 parts dry, 0.8 parts wet for a deeply submerged ambient vibe
    (source * 0.2) + (reverb * 0.8);
}.play;
)

// 2. Stop the sound
x.free;



//
//
//
//    OTeyPiano: synthetic piano sounds
//
// 1. Define the OteyPiano SynthDef
(
SynthDef(\piano, {
    arg freq = 440, amp = 0.5, pan = 0.0, gate = 1, rho = 1.0;
    var sig, env;

    // A standard envelope to free the synth node when done
    env = EnvGen.kr(Env.asr(0.001, 1, 0.1), gate, doneAction: 2);

    // OteyPiano generates a mono signal
    sig = OteyPiano.ar(freq, rho: rho, mul: amp);

    // Apply the envelope
    sig = sig * env;

    // FIX: Use Pan2 instead of Balance2 for mono-to-stereo panning
    sig = Pan2.ar(sig, pan);

    Out.ar(0, sig);
}).add;
)


// 2. Play a generative melody sequence
(
p = Pbind(
    \instrument, \piano,

    // Choose notes from a major 7th chord structure
    \degree, Pseq([[0, 2, 4, 7], 1, 4, 6, [2, 4, 6, 9], 3, 5, 7], inf),
    \scale, Scale.major,
    \octave, Pwhite(4, 6, inf).stutter(4), // Alternate octaves randomly

    // Durations and rhythm
    \dur, Prand([0.25, 0.5, 0.75], inf),

    // Random velocity / loudness
    \amp, Pwhite(0.2, 0.6, inf),

    // Subtle panning variation for each note
    \pan, Pwhite(-0.5, 0.5, inf),

    // 'rho' parameter control (higher values = brighter, longer ring)
    \rho, Pwhite(0.8, 1.5, inf),

    // Hold time for each note
    \legato, 2.0
).play;
)

// 3. Stop the sequence
p.stop;




//
//
//
//
// SmoothDecimator is a brilliant bitcrusher. Unlike standard harsh downsamplers, it includes a smoothing control (smooth) that rounds out the jagged digital edges, letting you dial in anything from a warm lo-fi crunch to absolute digital destruction
//
// 1. Play the SmoothDecimator example
(
x = {
    var source, fx;

    // A clean, simple sine sweep as our test source
    source = SinOsc.ar(XLine.kr(100, 2000, 10, doneAction: 2));

    fx = SmoothDecimator.ar(
        source,
        rate: MouseX.kr(1000, 44100, \exponential), // X-axis: Resample rate (lower = more crunchy)
        smoothing: MouseY.kr(0.0, 1.0),            // Y-axis: Smooth factor (0 = harsh, 1 = filtered/warm)
        mul: 0.3
    );

    fx ! 2; // Stereo duplicate
}.play;
)

x.free;



//
//
//
//
//
//
//Greyhole is a pitch-shifting, metallic, cloud-like reverb. It is structurally similar to Simbad's famous chorus-reverb algorithms. It specializes in giant, drifting metallic textures and infinite drones.
//
// 1. Play the Greyhole example
(
x = {
    var source, reverb;

    // A sparse source: simple impulses triggering short sine bursts
    source = Dust.ar(0.5);
    source = Decay.ar(source, 0.1) * SinOsc.ar(440);

    reverb = Greyhole.ar(
        source,
        delaytime: 0.2,                   // Base delay time
        damp: 0.1,                        // High frequency damping
        size: 3.5,                        // Size of the virtual space (larger = more diffusion)
        diff: 0.807,                      // Diffusion shape
        feedback: 0.9,                    // Decay time (values near 1.0 approach infinite ring)
        modDepth: MouseX.kr(0.0, 1.0),    // X-axis: Internal modulation depth (adds chorus/detuning)
        modFreq: 0.5                      // Modulation speed
    );

    // Mix dry and wet together
    (source * 0.1) + (reverb * 0.9) ! 2;
}.play;
)

x.free;



//
//
//
//
//
//
// ==========================================
// 1. ALLOCATE BUS AND DEFINE INSTRUMENTS
// ==========================================
(
// Create a 2-channel audio bus for routing
~fxBus = Bus.audio(s, 2);

// PIANO SOURCE: Outputs directly to the FX Bus instead of hardware output
SynthDef(\pianoSource, {
    arg freq = 440, amp = 0.5, gate = 1, rho = 1.0, out = 0, pan = 0.0;
    var sig, env;

    env = EnvGen.kr(Env.asr(0.001, 1, 0.2), gate, doneAction: 2);
    sig = OteyPiano.ar(freq, rho: rho, mul: amp) * env;

    // Pan the mono piano into the stereo FX bus
    Out.ar(out, Pan2.ar(sig, pan));
}).add;

// FX CHAIN: SmoothDecimator -> JPverb
SynthDef(\fxChain, {
    arg inBus, out = 0, rate = 12000, smoothing = 0.4, verbMix = 0.75;
    var dry, crushed, wet, final;

    // Read the stereo signal coming from the piano notes
    dry = In.ar(inBus, 2);

    // Step 1: SmoothDecimator (Degrade the digital quality)
    crushed = SmoothDecimator.ar(dry, rate: rate, smoothing: smoothing);

    // Step 2: JPverb (Wash the lo-fi texture into space)
    wet = JPverb.ar(
        crushed,
        t60: 8.0,          // 8-second long ambient tail
        damp: 0.4,         // Darker, warmer reflections
        size: 2.5,         // Giant cavernous space
        lowcut: 150,       // Prevent muddy low-end buildup
        highcut: 4500      // Roll off harsh high-frequency digital aliasing
    );

    // Combine the crushed signal and the wet reverb wash
    final = (crushed * (1 - verbMix)) + (wet * verbMix);

    Out.ar(out, final * 0.7); // Master volume safety scale
}).add;
)

// ==========================================
// 2. LIVE RUNTIME & SEQUENCER
// ==========================================
// ==========================================
// 2. LIVE RUNTIME & SEQUENCER (FIXED)
// ==========================================
(
// Instantiate the FX processor at the TAIL of the server line
~fxNode = Synth.tail(s, \fxChain, [\inBus, ~fxBus]);

// Play a slow, moody generative chord progression
~pattern = Pbind(
    \instrument, \pianoSource,
    \out, ~fxBus,               // Route notes through our FX bus

    // FIX: Removed the empty initial comma so SuperCollider can parse the array cleanly
    \midinote, Pseq([ // Cmaj7, // Bmin7, // Amin7
        [43, 50, 55, 59, 62]  // Gmaj7
    ], inf),

    \dur, 5.0,                // Change chords every 5 seconds
    \legato, 0.95,            // Let the chords overlap slightly
    \amp, Pwhite(0.15, 0.25, inf),
    \rho, Pwhite(0.6, 1.1, inf), // Randomizes hammer/string texture per note
    \pan, Pwhite(-0.3, 0.3, inf)
).play;
)


// ==========================================
// 3. STOP AND CLEAN UP
// ==========================================
(
~pattern.stop;
~fxNode.free;
)


// Make it intensely crushed and grainy:
~fxNode.set(\rate, 2500, \smoothing, 0.1, \verbMix, 0.6);

// Make it clean, dark, and highly submerged:
~fxNode.set(\rate, 44100, \smoothing, 0.9, \verbMix, 0.9);



//
//
//
//
//
//    VSTPlugin
//
// 1. Scan your system for available VST instruments and effects
// Scan all system-wide and user-specific audio folders natively
VSTPlugin.search(useSystem: true, useUser: true);
// Arg 1: server (nil defaults to the active server)
// Arg 2: dir (an array containing your custom VST3 paths)
VSTPlugin.search(s, ["/Library/Audio/Plug-Ins/VST3", "~/Library/Audio/Plug-Ins/VST3"]);
File.exists("/Library/Audio/Plug-Ins/VST3").postln;
// Print out the contents of the folder to your Post Window:
PathName("/Library/Audio/Plug-Ins/VST3").entries.do({|f| f.fileName.postln});

(
// Create a stereo audio routing bus
~vstBus = Bus.audio(s, 2);

// PIANO SOURCE: Outputs to our internal VST routing bus
SynthDef(\pianoToVst, {
    arg freq = 440, amp = 0.5, gate = 1, rho = 1.0, out = 0, pan = 0.0;
    var sig, env;
    env = EnvGen.kr(Env.asr(0.001, 1, 0.2), gate, doneAction: 2);
    sig = OteyPiano.ar(freq, rho: rho, mul: amp) * env;
    Out.ar(out, Pan2.ar(sig, pan));
}).add;

// VST HOST FX: Wraps the external audio stream and feeds it into the VST UGen
SynthDef(\supermassiveHost, { |out = 0, inBus = 0|
    var input, fx;
    input = In.ar(inBus, 2); // Read the stereo piano signal
    fx = VSTPlugin.ar(input, numOut: 2); // Process it via the VST
    Out.ar(out, fx);
}).add;
)

// gui user interface
(
// 1. Create the Host FX Synth at the tail of the server execution chain
~supermassiveSynth = Synth.tail(s, \supermassiveHost, [\inBus, ~vstBus]);

// 2. Assign the language controller to talk to the UGen
~supermassive = VSTPluginController(~supermassiveSynth);

// 3. Open the plugin file and display its visual GUI layout
~supermassive.open("ValhallaSupermassive", action: { |vst|
    "Valhalla Supermassive successfully loaded!".postln;
    vst.editor; // Opens the native Valhalla UI window!
});
)



(
~pianoPattern = Pbind(
    \instrument, \pianoToVst,
    \out, ~vstBus, // Push audio directly into Supermassive
    \midinote, Pseq([62, 66, 69, 73], inf), // E-minor 9th structures
    \dur, 0.75,
    \legato, 1.5,
    \amp, Pwhite(0.2, 0.4, inf),
    \rho, Pwhite(0.7, 1.2, inf),
    \pan, Pwhite(-0.4, 0.4, inf)
).play;
)


(
~pianoPattern.stop;
~supermassiveSynth.free;
~vstBus.free;
)