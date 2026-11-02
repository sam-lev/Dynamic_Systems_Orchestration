
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
