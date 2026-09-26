"""Convert one timestamped stereo capture into a calibrated dashboard observation.

Input NPZ comes from a sensor adapter, not two independent MJPEG downloads:
left_rgb, right_rgb, left_at, right_at, pose_at, pose_json, T_robot_left_camera.
Factory calibration JSON: image_size, K_left, K_right, dist_left, dist_right,
R_right_left, t_right_left_m, calibration_id, uncertainty_m.

python tools/stereo_depth.py capture.npz calibration.json observation.npz
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.perception import stereo_depth, finite_array


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path); ap.add_argument('calibration',type=Path); ap.add_argument('output',type=Path)
    a=ap.parse_args(); cal=json.loads(a.calibration.read_text())
    with np.load(a.capture,allow_pickle=False) as capture:
        rgb,depth,K,R1=stereo_depth(capture['left_rgb'],capture['right_rgb'],cal,float(capture['left_at']),float(capture['right_at']))
        T=finite_array(capture['T_robot_left_camera'],(4,4),'T_robot_left_camera').copy()
        # Rectification rotates original camera coordinates into rectified coordinates.
        T[:3,:3]=T[:3,:3]@R1.T
        temp=a.output.with_suffix('.tmp')
        with temp.open('wb') as out:
            np.savez_compressed(out,rgb=rgb,depth_m=depth,K=K,T_robot_camera=T,
                captured_at=float(capture['left_at']),pose_at=float(capture['pose_at']),pose_json=str(capture['pose_json']),
                calibration_id=cal['calibration_id'],uncertainty_m=float(cal.get('uncertainty_m',.02)))
        temp.replace(a.output)
    print(f'Wrote {a.output}: {np.isfinite(depth).mean():.1%} valid depth; capture timestamps preserved.')


if __name__=='__main__': main()
