"""Calibrated RGB/depth observations. No guessed distances or missing-depth fill.

NPZ input: rgb (H,W,3 uint8), depth_m (H,W), K (3,3), T_robot_camera (4,4),
captured_at, pose_at (Unix seconds), pose_json (scalar JSON), calibration_id.
Camera coordinates: x right, y down, z forward. Robot: x forward, y left, z up.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time
import zipfile
import numpy as np


class PerceptionError(ValueError):
    pass


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise PerceptionError(f'{name} must be finite with shape {shape}')
    return a


@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    transform: np.ndarray
    captured_at: float
    pose: dict
    calibration_id: str
    uncertainty: float = .02

    @classmethod
    def load(cls, path, max_age=3.0):
        path = Path(path)
        if path.stat().st_size > 64 * 1024 * 1024:
            raise PerceptionError('Observation exceeds 64 MB')
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
                raise PerceptionError('Uncompressed observation exceeds 64 MB')
        with np.load(path, allow_pickle=False) as pack:
            rgb = np.array(pack['rgb'])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not 16 <= min(rgb.shape[:2]) or max(rgb.shape[:2]) > 2048:
                raise PerceptionError('RGB must be uint8 H×W×3, between 16 and 2048 pixels per axis')
            depth = np.asarray(pack['depth_m'], dtype=float)
            if depth.shape != rgb.shape[:2]:
                raise PerceptionError('Depth must be aligned to RGB at the same resolution')
            K = finite_array(pack['K'], (3, 3), 'K')
            if min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8:
                raise PerceptionError('Unsupported camera intrinsics')
            T = finite_array(pack['T_robot_camera'], (4, 4), 'T_robot_camera')
            if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]), 1, atol=1e-5):
                raise PerceptionError('Camera transform must be rigid and right-handed')
            captured = float(pack['captured_at'])
            pose_at = float(pack['pose_at'])
            if not np.isfinite([captured, pose_at]).all() or not -.1 <= time.time() - captured <= max_age:
                raise PerceptionError('Depth observation is stale or has an invalid timestamp')
            if abs(captured - pose_at) > .05:
                raise PerceptionError('Robot pose and image differ by more than 50 ms')
            pose = json.loads(str(pack['pose_json']))
            if not isinstance(pose, dict) or not pose or not all(isinstance(k, str) and isinstance(v, (float, int)) and np.isfinite(v) for k, v in pose.items()):
                raise PerceptionError('Measured joint pose is required')
            calibration_id = str(pack['calibration_id'])
            if not calibration_id.strip():
                raise PerceptionError('Calibration ID is required')
            uncertainty = float(pack['uncertainty_m']) if 'uncertainty_m' in pack else .02
            if not np.isfinite(uncertainty) or not .005 <= uncertainty <= .05:
                raise PerceptionError('Uncertainty must be between 5 and 50 mm')
        depth = np.where(np.isfinite(depth) & (depth > .1) & (depth < 5), depth, np.nan)
        return cls(rgb, depth, K, T, captured, pose, calibration_id, uncertainty)

    def points(self, u, v, z):
        xyz = np.column_stack(((u - self.K[0,2]) * z / self.K[0,0],
                               (v - self.K[1,2]) * z / self.K[1,1], z))
        return xyz @ self.transform[:3,:3].T + self.transform[:3,3]

    def locate(self, bbox):
        """Robust central depth cluster, not the bounding-box centre's single pixel."""
        box = finite_array(bbox, (4,), 'Object box')
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise PerceptionError('Object box must be normalized xyxy')
        h, w = self.depth.shape
        x0, y0, x1, y1 = box * [w,h,w,h]
        # Inset avoids background mixing at uncertain image boundaries.
        dx, dy = (x1-x0)*.2, (y1-y0)*.2
        u, v = np.meshgrid(np.arange(int(x0+dx), max(int(x0+dx)+1,int(x1-dx))),
                           np.arange(int(y0+dy), max(int(y0+dy)+1,int(y1-dy))))
        z = self.depth[v, u]
        valid = np.isfinite(z)
        if valid.sum() < 20 or valid.mean() < .7:
            raise PerceptionError('Insufficient valid depth on the object; unknown space remains blocked')
        median = np.median(z[valid])
        cluster = valid & (np.abs(z-median) < max(.03, self.uncertainty*2))
        if cluster.sum() < 20 or cluster.sum() / valid.sum() < .7:
            raise PerceptionError('Object depth is ambiguous; acquire a clearer observation')
        pts = self.points(u[cluster], v[cluster], z[cluster])
        surface = np.median(pts, axis=0)
        outward = self.transform[:3,3] - surface
        outward /= np.linalg.norm(outward)
        return surface, outward, {'valid_fraction': round(float(valid.mean()),3),
                                  'depth_m': round(float(median),3), 'points': int(cluster.sum())}

    def require_free(self, points, radius):
        """Conservative depth-image free-space test for swept link samples.

        Checks the entire projected sphere rectangle. Occluded/out-of-view/invalid
        pixels are unknown and reject the path. It intentionally cannot certify
        an arm that the camera does not see.
        """
        cam = (np.asarray(points) - self.transform[:3,3]) @ self.transform[:3,:3]
        h,w = self.depth.shape
        for x,y,z in cam:
            if z <= radius + .1:
                raise PerceptionError('Moving link is outside the observed depth volume')
            u, v = self.K[0,0]*x/z+self.K[0,2], self.K[1,1]*y/z+self.K[1,2]
            # Bound perspective displacement including off-axis sphere centres.
            ru = self.K[0,0]*radius*(z+abs(x))/(z*(z-radius))
            rv = self.K[1,1]*radius*(z+abs(y))/(z*(z-radius))
            x0,x1,y0,y1 = int(np.floor(u-ru)),int(np.ceil(u+ru)),int(np.floor(v-rv)),int(np.ceil(v+rv))
            if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
                raise PerceptionError('Moving link leaves the camera field of view; free space is unknown')
            patch = self.depth[y0:y1+1,x0:x1+1]
            if not np.isfinite(patch).all():
                raise PerceptionError('Missing depth intersects the planned arm volume')
            if np.min(patch) <= z + radius + self.uncertainty:
                raise PerceptionError('Observed obstacle or occlusion intersects the planned arm volume')


def stereo_depth(left, right, calibration, left_at, right_at):
    """Rectify a calibrated stereo pair and return metric depth + rectified RGB/K.

    Driver timestamps must refer to capture, not MJPEG download times. No claim
    of hardware synchronization is made by this function.
    """
    import cv2
    if not np.isfinite([left_at,right_at]).all() or abs(left_at-right_at) > .015:
        raise PerceptionError('Stereo capture timestamps differ by more than 15 ms')
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise PerceptionError('Stereo images must have equal H×W×3 dimensions')
    h,w = left.shape[:2]
    if list(calibration['image_size']) != [w,h]:
        raise PerceptionError('Calibration resolution differs from stereo images')
    K1 = finite_array(calibration['K_left'],(3,3),'K_left')
    K2 = finite_array(calibration['K_right'],(3,3),'K_right')
    R = finite_array(calibration['R_right_left'],(3,3),'R_right_left')
    T = finite_array(calibration['t_right_left_m'],(3,),'t_right_left_m')
    if not .01 <= np.linalg.norm(T) <= .5 or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):
        raise PerceptionError('Invalid stereo baseline or rotation')
    D1,D2 = [np.asarray(calibration[k],float) for k in ('dist_left','dist_right')]
    if not all(np.isfinite(d).all() and d.size in (4,5,8,12,14) for d in (D1,D2)):
        raise PerceptionError('Invalid distortion coefficients')
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K1,D1,K2,D2,(w,h),R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    if abs(P2[1,3]) > abs(P2[0,3]) or P2[0,3] >= 0:
        raise PerceptionError('Expected horizontal stereo with the right camera to the right of the left')
    maps = [cv2.initUndistortRectifyMap(K,D,RR,P,(w,h),cv2.CV_32FC1) for K,D,RR,P in [(K1,D1,R1,P1),(K2,D2,R2,P2)]]
    l,r = [cv2.remap(im,*mp,cv2.INTER_LINEAR) for im,mp in zip((left,right),maps)]
    lg,rg = [cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in (l,r)]
    nd = min(128, ((w//3)//16)*16)
    if nd < 16:
        raise PerceptionError('Stereo image width is too small')
    def matcher(minimum):
        return cv2.StereoSGBM_create(minDisparity=minimum,numDisparities=nd,blockSize=5,P1=8*25,P2=32*25,
                                    uniquenessRatio=12,speckleWindowSize=80,speckleRange=2,disp12MaxDiff=1)
    disparity = matcher(0).compute(lg,rg).astype(float)/16
    reverse = matcher(-nd).compute(rg,lg).astype(float)/16
    yy,xx = np.indices((h,w)); xr = np.rint(xx-disparity).astype(int)
    valid = (disparity > 0) & (xr >= 0) & (xr < w)
    valid &= np.abs(disparity + reverse[yy,np.clip(xr,0,w-1)]) < 1.5
    xyz = cv2.reprojectImageTo3D(disparity.astype(np.float32),Q)
    depth = np.where(valid & (xyz[:,:,2] > .1) & (xyz[:,:,2] < 5),xyz[:,:,2],np.nan)
    return l, depth, P1[:3,:3], R1
