"""从录像离线标定相机在云台本体上的安装（ArmorTracker 的 mount_rotation_wxyz /
mount_translation）。

Offline calibration of the camera mounting on the gimbal body (ArmorTracker's
mount_rotation_wxyz / mount_translation) from a recording.

GShang 板固定不动；云台转到多个不同姿态，每个姿态停住约半秒，录像带 IMU 列。工具按陀螺
找出静止段，每段取中间一帧，用内参求板的位姿，再与 IMU 姿态一起解手眼：
The board stays fixed; the gimbal visits several attitudes and holds each for about
half a second, recorded with the IMU columns. The tool finds still segments from the
gyro, takes the middle frame of each, solves the board pose with the intrinsics and
solves hand-eye together with the IMU attitudes:

  python calibrate_handeye.py REC_DIR --intrinsics intrinsics.yaml --out mount.yaml

云台绕固定点转动时平移只能粗略确定；旋转是主要结果。
With the gimbal rotating about a fixed point the translation is only roughly
determined; the rotation is the main result.
"""
import argparse
import sys

import cv2
import numpy as np

import recording

# 光学系（x 右、y 下、z 前）到本体系（x 右、y 前、z 上）/ Optical to body axes.
OPTICAL_TO_BODY = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)


def quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(r):
    """wxyz，w ≥ 0。"""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def load_intrinsics(path):
    values = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    k = np.array([[float(values["fx"]), 0, float(values["cx"])],
                  [0, float(values["fy"]), float(values["cy"])], [0, 0, 1]])
    dist = np.array([float(x) for x in values["distortion"].strip("[]").split(",")])
    return k, dist


def still_frames(rows, still_rad_s, min_frames):
    """陀螺模长持续低于阈值的段，每段返回中间一帧。"""
    picked, run = [], []
    for row in rows + [None]:
        if row is not None and np.linalg.norm(row.gyro) < still_rad_s:
            run.append(row)
            continue
        if len(run) >= min_frames:
            picked.append(run[len(run) // 2])
        run = []
    return picked


def angle_deg(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="recording directory with IMU columns")
    ap.add_argument("--intrinsics", required=True, help="YAML from calibrate_intrinsics.py")
    ap.add_argument("--marker-mm", type=float, default=25.0)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--min-corners", type=int, default=12, help="chessboard corners per view")
    ap.add_argument("--still-rad-s", type=float, default=0.02)
    ap.add_argument("--min-still-frames", type=int, default=5)
    ap.add_argument("--out", default="mount.yaml")
    a = ap.parse_args(argv)
    k, dist = load_intrinsics(a.intrinsics)
    board = recording.gshang_board(a.marker_mm, a.cols, a.rows)
    detector = recording.make_detector()

    rows = recording.load_frames(a.dir)
    if not rows or rows[0].gyro is None:
        sys.exit("the recording has no IMU columns")
    r_board2cam, t_board2cam, r_world2body, points = [], [], [], []
    for row in still_frames(rows, a.still_rad_s, a.min_still_frames):
        found = recording.detect_board(detector, recording.read_gray(a.dir, row.frame), board)
        if found is None or len(found[0]) < a.min_corners:
            continue
        obj, img = found
        native = recording.frame_to_native(img, row)
        ok, rvec, tvec = cv2.solvePnP(obj, native, k, dist)
        if not ok:
            continue
        r_board2cam.append(cv2.Rodrigues(rvec)[0])
        t_board2cam.append(tvec.reshape(3, 1))
        r_world2body.append(quat_to_matrix(row.quat_wxyz).T)
        points.append((obj, native))
    if len(r_board2cam) < 3:
        sys.exit(f"only {len(r_board2cam)} still poses with the board; need at least 3")
    spread = max(angle_deg(r_world2body[i] @ r_world2body[j].T)
                 for i in range(len(r_world2body)) for j in range(i))
    if spread < 5.0:
        sys.exit(f"gimbal attitudes span only {spread:.1f} deg; move further between poses")

    # 板 = OpenCV 的 world，IMU 世界系 = base，本体 = gripper
    # The board is OpenCV's world, the IMU world is the base, the body is the gripper.
    zeros = [np.zeros((3, 1))] * len(r_world2body)
    r_base2world, t_base2world, r_body2cam, t_body2cam = cv2.calibrateRobotWorldHandEye(
        r_board2cam, t_board2cam, r_world2body, zeros)
    # 初值换成板到 IMU 世界系、相机到本体 / Initial board-to-IMU-world and camera-to-body.
    r_board2imu = r_base2world.T
    x0 = np.concatenate([cv2.Rodrigues(r_board2imu)[0].ravel(),
                         (-r_board2imu @ t_base2world).ravel(),
                         cv2.Rodrigues(r_body2cam.T)[0].ravel(),
                         (-r_body2cam.T @ t_body2cam).ravel()])
    r_body2world = [r.T for r in r_world2body]
    rms0 = np.sqrt(np.mean(reprojection(x0, r_body2world, points, k, dist) ** 2))
    x = refine(x0, r_body2world, points, k, dist)
    rms = np.sqrt(np.mean(reprojection(x, r_body2world, points, k, dist) ** 2))
    r_cam2body = cv2.Rodrigues(x[6:9])[0]
    t_cam2body = x[9:12]
    q = matrix_to_quat(r_cam2body @ OPTICAL_TO_BODY.T)

    with open(a.out, "w") as f:
        f.write(f"# {len(points)} poses spanning {spread:.1f} deg, reprojection RMS "
                f"{rms:.3f} px (hand-eye start {rms0:.3f} px)\n")
        f.write("mount_rotation_wxyz: [" + ", ".join(f"{v:.8f}" for v in q) + "]\n")
        f.write("mount_translation: [" + ", ".join(f"{v:.4f}" for v in t_cam2body) + "]\n")
    print(f"{len(points)} poses, reprojection RMS {rms:.3f} px (start {rms0:.3f}) -> {a.out}")
    print("mount_rotation_wxyz:", np.round(q, 8).tolist())
    print("mount_translation:", np.round(t_cam2body, 4).tolist())
    return q, t_cam2body, rms


def reprojection(x, r_body2world, points, k, dist):
    """参数 x = 板到世界的旋转向量与平移、相机到本体的旋转向量与平移；返回所有姿态的像素残差。

    x holds the board-to-world rotation vector and translation and the camera-to-body
    rotation vector and translation; returns the pixel residuals of all poses.
    """
    r_b2w, t_b2w = cv2.Rodrigues(x[0:3])[0], x[3:6]
    r_c2b, t_c2b = cv2.Rodrigues(x[6:9])[0], x[9:12]
    out = []
    for r_bw, (obj, native) in zip(r_body2world, points):
        r_c2w = r_bw @ r_c2b
        r_board2cam = r_c2w.T @ r_b2w
        t_board2cam = r_c2w.T @ (t_b2w - r_bw @ t_c2b)
        p, _ = cv2.projectPoints(obj, cv2.Rodrigues(r_board2cam)[0], t_board2cam, k, dist)
        out.append((p.reshape(-1, 2) - native).ravel())
    return np.concatenate(out)


def refine(x, r_body2world, points, k, dist, iterations=50):
    """所有姿态的重投影误差一起做 Levenberg-Marquardt；单个姿态 PnP 的倾角误差（约 1°）
    在这里被平均掉。

    Levenberg-Marquardt over the reprojection error of all poses; the out-of-plane tilt
    error of single-pose PnP (about 1 deg) averages out here.
    """
    f = lambda v: reprojection(v, r_body2world, points, k, dist)
    r = f(x)
    lam = 1e-3
    for _ in range(iterations):
        j = np.empty((r.size, x.size))
        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = 1e-6
            j[:, i] = (f(x + step) - r) / 1e-6
        jtj, jtr = j.T @ j, j.T @ r
        while True:
            delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj)), -jtr)
            r_new = f(x + delta)
            if r_new @ r_new < r @ r:
                x, r, lam = x + delta, r_new, lam * 0.3
                break
            lam *= 10
            if lam > 1e8:
                return x
        if np.linalg.norm(delta) < 1e-10:
            break
    return x


if __name__ == "__main__":
    main()
