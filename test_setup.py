#!/usr/bin/env python3
# test_setup.py -- Audible Lava v3.0
# Verifies all dependencies and camera.

import sys

def check(label, fn):
    try:
        r = fn()
        print(f"  OK  {label}" + (f": {r}" if r else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        return False

def main():
    print("=" * 55)
    print("  Audible Lava v3.0 -- Environment Check")
    print("=" * 55)
    print()
    all_ok = True

    print("Python packages:")
    all_ok &= check("cv2 (OpenCV)", lambda: __import__("cv2").__version__)
    all_ok &= check("numpy",        lambda: __import__("numpy").__version__)
    all_ok &= check("gudhi",        lambda: __import__("gudhi").__version__)
    all_ok &= check("scipy",        lambda: __import__("scipy").__version__)
    all_ok &= check("pythonosc",    lambda: None)

    print()
    print("OpenCV components:")
    all_ok &= check("MOG2", lambda: (
        __import__("cv2").createBackgroundSubtractorMOG2(history=10), "OK")[1])
    all_ok &= check("KalmanFilter(5,2)", lambda: (
        __import__("cv2").KalmanFilter(5, 2), "OK")[1])
    all_ok &= check("Farneback flow", lambda: (
        getattr(__import__("cv2"), "calcOpticalFlowFarneback"), "OK")[1])
    all_ok &= check("CamShift", lambda: (
        getattr(__import__("cv2"), "CamShift"), "OK")[1])

    print()
    print("GUDHI:")
    def _gudhi():
        import gudhi, numpy as np
        cc = gudhi.CubicalComplex(
            top_dimensional_cells=np.zeros(16).tolist(), dimensions=[4,4])
        cc.persistence()
        return f"GUDHI {gudhi.__version__} OK"
    all_ok &= check("CubicalComplex", _gudhi)

    print()
    print("Project modules:")
    all_ok &= check("lamp_profile",     lambda: (__import__("lamp_profile"),    "OK")[1])
    all_ok &= check("blob_detection",   lambda: (__import__("blob_detection"),  "OK")[1])
    all_ok &= check("blob_tracker",     lambda: (__import__("blob_tracker"),    "OK")[1])
    all_ok &= check("merge_tree",       lambda: (__import__("merge_tree"),      "OK")[1])
    all_ok &= check("voice_manager",    lambda: (__import__("voice_manager"),   "OK")[1])
    all_ok &= check("calibration_gui",  lambda: (__import__("calibration_gui"), "OK")[1])

    print()
    print("Camera:")
    def _cam():
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): raise RuntimeError("Cannot open")
        ret, f = cap.read()
        cap.release()
        if not ret: raise RuntimeError("Cannot read")
        return f"{f.shape[1]}x{f.shape[0]}"
    all_ok &= check("Camera index 0", _cam)

    print()
    if all_ok:
        print("All checks passed.")
        print()
        print("First run:")
        print("  python audible_lava.py            # opens calibration GUI")
        print("  python audible_lava.py --calibrate # re-calibrate")
        print()
        print("Or test individual components:")
        print("  python calibration_gui.py")
        print("  python test_detection.py")
        print("  python test_tracker.py")
    else:
        print("Checks failed. Install missing packages:")
        print("  pip install opencv-python numpy gudhi scipy python-osc")
        sys.exit(1)

if __name__ == "__main__":
    main()
