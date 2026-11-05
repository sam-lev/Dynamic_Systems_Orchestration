# RAVE bench report

models dir: /Users/multivax/Documents/RAVE_PreTrained_Models
out: rave_bench_out

- models: ['organ', 'water', 'vocalset']
- buffers: [8192]
- quick: True
- master: True
- sc_rate: 48000
- nn_available: True
- scenarios: 23

| scenario | group | stim | models | buf | gain | rms dB | peak | silent % | dropouts | clicks | latency ms | cpu proc % | dsp % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| split_passthrough | functional | tone | - | 2048 | 1.0 | -24.1 | -19.2 | 2 | 1 (50ms) | 0 | - | 10.3 | 9.3 |  |
| tone_reference | reference | tone | - | 2048 | 1.0 | -24.0 | -19.2 | 2 | 1 (50ms) | 0 | - | 10.2 | 8.2 |  |
| dry_scaling | functional | tone | - | 2048 | 1.0 | -34.5 | -29.7 | 2 | 2 (20ms) | 0 | - | 9.0 | 7.5 | PASS |
| burst_reference | reference | burst | - | 2048 | 1.0 | -39.6 | -15.7 | 93 | 0 (0ms) | 0 | - | 9.9 | 9.2 |  |
| lamp_reference | reference | lamp | - | 2048 | 1.0 | -20.1 | -6.4 | 6 | 0 (0ms) | 0 | - | 21.1 | 19.6 |  |
| iso_drones | isolation | lamp:drones | - | 2048 | 1.0 | -22.5 | -8.3 | 6 | 0 (0ms) | 0 | - | 13.9 | 12.7 |  |
| iso_pluck | isolation | lamp:pluck | - | 2048 | 1.0 | -39.4 | -16.1 | 80 | 0 (0ms) | 0 | - | 11.7 | 11.1 |  |
| iso_piano | isolation | lamp:piano | - | 2048 | 1.0 | -31.8 | -16.0 | 79 | 0 (0ms) | 0 | - | 11.0 | 8.6 |  |
| iso_bowed | isolation | lamp:bowed | - | 2048 | 1.0 | -24.9 | -16.2 | 64 | 0 (0ms) | 0 | - | 10.5 | 8.4 |  |
| iso_plucknote | isolation | lamp:plucknote | - | 2048 | 1.0 | -41.1 | -17.5 | 89 | 0 (0ms) | 0 | - | 8.4 | 8.4 |  |
| iso_strike | isolation | lamp:strike | - | 2048 | 1.0 | -46.2 | -25.7 | 83 | 0 (0ms) | 0 | - | 11.5 | 10.0 |  |
| iso_bass | isolation | lamp:bass | - | 2048 | 1.0 | -42.0 | -22.0 | 72 | 0 (0ms) | 0 | - | 10.9 | 10.1 |  |
| organ_b8192_tone | model_on | tone | organ | 8192 | 1.0 | -26.1 | -13.3 | 0 | 0 (0ms) | 0 | - | 48.3 | 7.8 | wet ok; no dropouts |
| organ_b8192_burst | latency | burst | organ | 8192 | 1.0 | -43.4 | -13.8 | 91 | 0 (0ms) | 0 | - | 49.8 | 8.4 | wet ok |
| organ_b8192_lamp | lamp | lamp | organ | 8192 | 1.0 | -24.5 | -4.0 | 7 | 2 (135ms) | 0 | - | 60.5 | 17.2 | wet ok; DROPOUTS 2 (135 ms); GAPPY 7% silent |
| water_b8192_tone | model_on | tone | water | 8192 | 1.0 | -27.8 | -15.1 | 0 | 0 (0ms) | 0 | - | 48.0 | 7.5 | wet ok; no dropouts |
| water_b8192_burst | latency | burst | water | 8192 | 1.0 | -54.0 | -27.9 | 92 | 0 (0ms) | 0 | - | 50.0 | 7.1 | wet ok |
| water_b8192_lamp | lamp | lamp | water | 8192 | 1.0 | -25.0 | -5.7 | 5 | 0 (0ms) | 0 | - | 62.4 | 16.7 | wet ok; no dropouts |
| vocalset_b8192_tone | model_on | tone | vocalset | 8192 | 1.0 | -29.7 | -22.9 | 0 | 0 (0ms) | 0 | - | 52.6 | 8.0 | wet ok; no dropouts |
| vocalset_b8192_burst | latency | burst | vocalset | 8192 | 1.0 | -64.2 | -37.0 | 96 | 0 (0ms) | 0 | - | 46.4 | 8.4 | silent |
| vocalset_b8192_lamp | lamp | lamp | vocalset | 8192 | 1.0 | -27.5 | -8.7 | 4 | 0 (0ms) | 0 | - | 54.8 | 15.5 | wet ok; no dropouts |
| stack2_b8192_lamp | stack | lamp | organ,water | 8192 | 1.0 | -25.9 | -8.2 | 6 | 0 (0ms) | 0 | - | 125.6 | 14.8 | wet ok; no dropouts; GAPPY 6% silent |
| stack3_b8192_lamp | stack | lamp | organ,water,vocalset | 8192 | 1.0 | -27.3 | -9.8 | 4 | 0 (0ms) | 0 | - | 300.1 | 18.2 | wet ok; no dropouts |

## Per model (first buffer)

| model | wet rms dB | latency ms | dropouts/s | cpu proc % | silence-in dB | gain where rms > -30 dB |
|---|---|---|---|---|---|---|
| organ | -26.1 | - | 0.0 | 48.3 | - | 1.0 |
| vocalset | -29.7 | - | 0.0 | 52.6 | - | 1.0 |
| water | -27.8 | - | 0.0 | 48.0 | - | 1.0 |

## Reading the numbers

- **iso_<part>** = one part of the emulated lamp run alone, no RAVE. A BLOWUP there names the SynthDef that goes unstable.
- **BLOWUP** = NaN/inf or samples over full scale in the recording. In a *_reference row it is the stimulus SynthDefs themselves; fix that first, because a RAVE model fed it is corrupted for the rest of its life (the v4.11.2 \raveSplit now sanitises and limits its feed to stop that).
- **silent %** = share of 20 ms frames below -60 dB after the model's load time, on a steady input. Above ~5 % the wet signal is audibly choppy.
- **dropouts** in a tone/lamp scenario = inference not keeping up (underruns). Compare the same model across buffers; if 8192 is clean and 2048 is not, raise ~raveBufferSize.
- **cpu proc %** is the whole scsynth process (inference threads included); **dsp %** is s.avgCPU, DSP thread only. A model can peg the machine while dsp % stays low.
- **latency** = burst-train delay dry->wet, shown only when the cross-correlation is unambiguous (confidence >= 0.3; raw value in the JSON).
- **silence-in dB** = what the model emits when fed nothing. Above -50 dB it hallucinates; keep such models off when the lamp is quiet.
- **gain where rms > -30 dB** = the input gain at which the model starts producing a useful level from a -10 dBFS tone; that is where to set RAVE_IN_GAIN.
- **stackN**: level should not grow with N (mix/sqrt n); dropouts appearing only at N>1 means one model at a time is the machine's limit.
