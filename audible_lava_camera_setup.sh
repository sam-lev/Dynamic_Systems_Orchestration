#!/bin/bash
# audible_lava_camera_setup.sh — locks C920 into deterministic state.
# Run with no other apps using the camera.

set -e   # exit on any uvcc error

# Verify uvcc is installed and the camera is visible
if ! command -v uvcc &> /dev/null; then
    echo "ERROR: uvcc not installed. Run: brew install uvcc"
    exit 1
fi

if ! uvcc devices | grep -qi "logitech\|046d"; then
    echo "ERROR: No Logitech camera detected. Plug in the C920."
    uvcc devices
    exit 1
fi

echo "Setting C920 to manual mode..."

# Set values; ignore individual failures so we see all errors at once
uvcc set auto_exposure_mode 1                   2>&1 || echo "  WARN: auto_exposure_mode failed"
uvcc set absolute_exposure_time 333             2>&1 || echo "  WARN: absolute_exposure_time failed"
uvcc set gain 0                                 2>&1 || echo "  WARN: gain failed"
uvcc set auto_white_balance_temperature 0       2>&1 || echo "  WARN: auto_white_balance_temperature failed"
uvcc set white_balance_temperature 3000         2>&1 || echo "  WARN: white_balance_temperature failed"
uvcc set saturation 128                         2>&1 || echo "  WARN: saturation failed"
uvcc set sharpness 0                            2>&1 || echo "  WARN: sharpness failed"
uvcc set focus_auto 0                           2>&1 || echo "  WARN: focus_auto failed"

echo ""
echo "Verifying values stuck:"
echo "  auto_exposure_mode:              $(uvcc get auto_exposure_mode)"
echo "  absolute_exposure_time:          $(uvcc get absolute_exposure_time)"
echo "  gain:                            $(uvcc get gain)"
echo "  auto_white_balance_temperature:  $(uvcc get auto_white_balance_temperature)"
echo "  white_balance_temperature:       $(uvcc get white_balance_temperature)"
echo "  saturation:                      $(uvcc get saturation)"
echo "  sharpness:                       $(uvcc get sharpness)"
echo ""
echo "Done. Launch the pipeline now without unplugging the camera."#!/bin/bash
# audible_lava_camera_setup.sh — run before launching the pipeline.
# Locks the C920 into deterministic, calibration-stable state.

uvcc set auto_exposure_mode 1                  # manual exposure
uvcc set absolute_exposure_time 333            # ~1/30s
uvcc set gain 0                                # no amplification
uvcc set auto_white_balance_temperature 0      # manual WB
uvcc set white_balance_temperature 3000        # tungsten bulb
uvcc set brightness 128                        # default
uvcc set contrast 128                          # default
uvcc set saturation 128                        # default — critical
uvcc set sharpness 0                           # no edge enhancement
uvcc set focus_auto 0                          # autofocus off
uvcc set focus_absolute 0                      # focus at infinity for fixed setup

echo "Camera locked. Verify with: uvcc get <param>"
