"""从录像离线标定相机内参（原生像素，plumb_bob 5 个系数）。

Offline intrinsic calibration from recordings (native pixels, five plumb_bob
coefficients).

录一段在相机前移动 GShang 标定板的录像（WIDE、NARROW 都可以，几何按每帧换算到原生
坐标），然后：
Record the GShang board moving in front of the camera (WIDE or NARROW, each frame is
mapped to native coordinates by its geometry), then:

  python calibrate_intrinsics.py REC_DIR [REC_DIR ...] --out intrinsics.yaml

输出 YAML 与可直接粘贴的 C++ 初始化式（CameraTypes::CameraCalibration）。
Writes YAML and a C++ initializer for CameraTypes::CameraCalibration.
"""
import argparse
import sys

import cv2
import numpy as np

import recording


def board_normal(obj, native, size):
    """用粗略内参（焦距取 1.5 倍长边）估计板法向，只用来区分视角。"""
    f = 1.5 * max(size)
    k = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    ok, rvec, _ = cv2.solvePnP(obj, native, k, None)
    return cv2.Rodrigues(rvec)[0][:, 2] if ok else np.array([0.0, 0.0, 1.0])


def collect(dirs, board, every, min_corners, size):
    """每个可用帧：(物体点, 原生像素, 板中心, 板尺度, 板法向)。"""
    detector = recording.make_detector()
    views = []
    for d in dirs:
        for row in recording.load_frames(d)[::every]:
            found = recording.detect_board(detector, recording.read_gray(d, row.frame), board)
            if found is None or len(found[0]) < min_corners:
                continue
            obj, img = found
            native = recording.frame_to_native(img, row)
            span = native.max(axis=0) - native.min(axis=0)
            views.append((obj, native, native.mean(axis=0), float(np.hypot(*span)),
                          board_normal(obj, native, size)))
    return views


def select(views, max_views, diagonal):
    """去掉与已选视角几乎相同的（中心差 < 4% 对角线、尺度差 < 10% 且倾斜差 < 8°），
    再均匀抽到上限。
    Drop views almost equal to a kept one (centre within 4% of the diagonal, scale
    within 10% and tilt within 8 deg), then subsample evenly to the limit.
    """
    kept = []
    cos_tilt = np.cos(np.radians(8.0))
    for v in views:
        if all(np.linalg.norm(v[2] - k[2]) > 0.04 * diagonal or abs(v[3] / k[3] - 1) > 0.1
               or v[4] @ k[4] < cos_tilt for k in kept):
            kept.append(v)
    if len(kept) > max_views:
        kept = [kept[i] for i in np.linspace(0, len(kept) - 1, max_views).round().astype(int)]
    return kept


def calibrate(views, size, flags=0):
    """返回 RMS、内参、5 个畸变系数、每个视角的 RMS、fx fy cx cy 的标准差。"""
    obj = [v[0].astype(np.float32) for v in views]
    img = [v[1].astype(np.float32) for v in views]
    rms, k, dist, rvecs, tvecs, std, _, _ = cv2.calibrateCameraExtended(
        obj, img, size, None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, k, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
    return rms, k, dist.ravel()[:5], np.array(per_view), std.ravel()[:4]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="recording directories")
    ap.add_argument("--native", default="1440x1080", help="native sensor size WxH")
    ap.add_argument("--marker-mm", type=float, default=25.0)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--every", type=int, default=2, help="use every N-th frame")
    ap.add_argument("--min-corners", type=int, default=12, help="chessboard corners per view")
    ap.add_argument("--max-views", type=int, default=80)
    ap.add_argument("--fix-k3", action="store_true", help="keep k3 at 0 (narrow lenses)")
    ap.add_argument("--out", default="intrinsics.yaml")
    a = ap.parse_args(argv)
    w, h = (int(x) for x in a.native.lower().split("x"))
    board = recording.gshang_board(a.marker_mm, a.cols, a.rows)

    views = collect(a.dirs, board, a.every, a.min_corners, (w, h))
    views = select(views, a.max_views, np.hypot(w, h))
    if len(views) < 10:
        sys.exit(f"only {len(views)} distinct views with the board; record more poses")
    flags = cv2.CALIB_FIX_K3 if a.fix_k3 else 0
    rms, k, dist, per_view, std = calibrate(views, (w, h), flags)
    # 去掉误差大于中位数 3 倍的视角再算一次 / Drop views above 3x the median, solve again.
    good = per_view <= 3 * np.median(per_view)
    if not good.all():
        views = [v for v, g in zip(views, good) if g]
        rms, k, dist, per_view, std = calibrate(views, (w, h), flags)

    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    d = ", ".join(f"{x:.8g}" for x in dist)
    summary = (f"{len(views)} views, RMS {rms:.3f} px, worst view {per_view.max():.3f} px, "
               f"sd fx {std[0]:.1f} fy {std[1]:.1f} cx {std[2]:.1f} cy {std[3]:.1f} px")
    with open(a.out, "w") as f:
        f.write(f"# {summary}\n")
        f.write(f"native_width: {w}\nnative_height: {h}\n")
        f.write(f"fx: {fx:.6f}\nfy: {fy:.6f}\ncx: {cx:.6f}\ncy: {cy:.6f}\n")
        f.write(f"distortion: [{d}]\n")
    print(f"{summary} -> {a.out}")
    print(f"{{{w}, {h}, {fx:.6f}, {fy:.6f}, {cx:.6f}, {cy:.6f}, {{{d}}}}}")
    # 焦距标准差超过 0.5%：视角不够多样，测距会跟着偏 / Focal sd above 0.5%: the views
    # lack variety and ranging will be off by as much.
    if std[0] > 0.005 * fx:
        print("WARNING: focal length uncertain; add views with the board tilted 30-45 deg")
    return fx, fy, cx, cy, dist, rms


if __name__ == "__main__":
    main()
