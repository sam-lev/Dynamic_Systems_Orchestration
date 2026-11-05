Server.killAll;
//• Go to the top menu bar and click Language -> Reboot Interpreter.
//• Once the post window completes the reboot process, try running s.boot; again.
s.options.numInputBusChannels = 0;
s.options.outDevice = "Pebble V3";
s.reboot;

//for plugins
Platform.systemExtensionDir
//SynthDescLib.global.synthDescs.keys.includes(\melodyHit)
//~wtBuffers[0].bufnum;
//Synth(\melodyHit, [\freq, 330, \amp, 0.4, \dur, 0.8, \bufBase, ~wtBuffers[0].bufnum]);
s.queryAllNodes;

s.sendMsg("/g_new", 1, 0, 0);           // recreate the default group
Synth(\melodyHit, [\freq, 330, \amp, 0.4, \dur, 0.8, \release, 1.0]);
Synth(\melodyHit, [\freq, 330, \amp, 0.4, \dur, 0.8, \bufBase, 1]);
Synth(\mpeVoice,  [\freq, 330, \amp, 0.3]).release(2);   // control: does the drone synth sound by hand?

Synth(\melodyHit, [\freq, 330, \amp, 0.4, \dur, 0.8, \release, 1.0]);          // bufBase 0, as Python sends it
Synth(\melodyHit, [\freq, 330, \amp, 0.4, \dur, 0.8, \release, 1.0, \bufBase, 1]);

{ SinOsc.ar(330, 0, 0.2) ! 2 }.play;                 // 1. output path
{ VOsc.ar(1, 330, mul: 0.3) ! 2 }.play;              // 2. wavetable buffers 1+2 directly
~wtBuffers[0].getn(0, 8, { |d| d.postln });          // 3. is buffer 1 filled? (non-zero numbers)
Synth(\mpeVoice, [\freq, 330, \amp, 0.3, \bufBase, 1]).release(2);   // 4. the drone synth, correct base
s.sendMsg("/g_new", 1, 0, 0);
{ VOsc.ar(1, 330, mul: 0.3) ! 2 }.play;
~wtBuffers[0].getn(0, 8, { |d| d.postln });

~wtBuffers[0].sine1((1..24).reciprocal);
~wtBuffers[1].sine1((1..24).collect { |i| ((i + 1) % 2 == 0).if(0, i.reciprocal) });
{ VOsc.ar(1, 330, mul: 0.3) ! 2 }.play;        // should now sound; Cmd+. to stop