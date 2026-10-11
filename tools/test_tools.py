"""离线标定工具的合成检查：按已知内参与安装渲染标定板录像，再用工具解回来。

Synthetic check of the offline calibration tools: render board recordings with known
intrinsics and mounting, then solve them back with the tools.

  python test_tools.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

import calibrate_handeye
import calibrate_intrinsics
import recording

NATIVE = (1440, 1080)
K = np.array([[2340.0, 0, 715.0], [0, 2336.0, 530.0], [0, 0, 1]])
WIDE = (80, 24, 2)
NARROW = (400, 284, 1)
# 码元 25/7 mm 正好 14 个纹理像素，marker 位置不必取整 / One code cell is exactly 14 texels.
PX_PER_M = 3920.0


def board_texture(board):
    """白底上画 GShang 板：黑格与白格中的 marker；纹理像素与板坐标按 PX_PER_M 对应。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    sq = int(round(board.square_m * PX_PER_M))
    tex = np.full((board.rows * sq + 40, board.cols * sq + 40), 255, np.uint8)
    for r in range(board.rows):
        for c in range(board.cols):
            if (r + c) % 2 == 0:
                tex[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
    for marker_id, quad in board.markers.items():
        x0, y0 = (quad[0, :2] * PX_PER_M).round().astype(int)
        side = int(round((quad[1, 0] - quad[0, 0]) * PX_PER_M))
        tex[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    return tex


def frame_camera(geometry):
    """帧像素的内参：x_f = (x_n - roi - phase) / decimation。"""
    roi_x, roi_y, dec = geometry
    phase = -0.5 if dec == 2 else 0.0
    return np.array([[K[0, 0] / dec, 0, (K[0, 2] - roi_x - phase) / dec],
                     [0, K[1, 1] / dec, (K[1, 2] - roi_y - phase) / dec], [0, 0, 1]])


def render(tex, r_board2cam, t_board2cam, geometry, ss=4):
    """按 ss 倍分辨率渲染再缩小（近似像素积分）；纹理像素中心 i 对应板上 (i + 0.5) / PX_PER_M。

    Render at ss times the resolution and shrink (approximates pixel integration); the
    centre of texel i is board coordinate (i + 0.5) / PX_PER_M.
    """
    kf = frame_camera(geometry)
    # 放大 ss 倍后像素中心 j 对应原像素 (j + 0.5) / ss - 0.5 / Pixel centres after scaling.
    up = np.array([[ss, 0, 0.5 * (ss - 1)], [0, ss, 0.5 * (ss - 1)], [0, 0, 1]])
    plane = np.column_stack([r_board2cam[:, 0], r_board2cam[:, 1], t_board2cam])
    texel = np.array([[1 / PX_PER_M, 0, 0.5 / PX_PER_M], [0, 1 / PX_PER_M, 0.5 / PX_PER_M],
                      [0, 0, 1]])
    big = cv2.warpPerspective(tex, up @ kf @ plane @ texel,
                              (recording.FRAME_W * ss, recording.FRAME_H * ss),
                              flags=cv2.INTER_LINEAR, borderValue=128)
    return cv2.resize(big, (recording.FRAME_W, recording.FRAME_H), interpolation=cv2.INTER_AREA)


def write_recording(directory, frames):
    """frames: [(gray, geometry, quat or None, gyro or None)]，灰度直接当 Bayer 写。"""
    os.makedirs(directory, exist_ok=True)
    imu = frames[0][2] is not None
    with open(os.path.join(directory, "frames.csv"), "w") as f:
        f.write("frame,timestamp_us,frame_counter,roi_x,roi_y,decimation")
        f.write(",qw,qx,qy,qz,gx,gy,gz,ax,ay,az\n" if imu else "\n")
        for i, (gray, (rx, ry, dec), q, g) in enumerate(frames):
            cv2.imwrite(os.path.join(directory, f"{i:06d}.pgm"), gray)
            f.write(f"{i},{i * 10000},{i},{rx},{ry},{dec}")
            if imu:
                f.write("," + ",".join(f"{x:.9g}" for x in [*q, *g, 0, 0, 9.8]))
            f.write("\n")


def rot(axis, deg):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.radians(deg)
    return cv2.Rodrigues(v)[0]


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    return ok


def test_intrinsics(tmp, tex, board):
    rng = np.random.default_rng(1)
    centre = np.concatenate(list(board.markers.values())).mean(axis=0)
    frames = []
    for i in range(60):
        geometry = NARROW if i % 4 == 3 else WIDE
        r = rot("x", rng.uniform(-35, 35)) @ rot("y", rng.uniform(-35, 35)) @ rot("z", rng.uniform(-20, 20))
        z = rng.uniform(0.5, 1.0) if geometry == WIDE else rng.uniform(1.0, 1.5)
        # 板中心落在画面里的随机位置 / The board centre at a random place in the view.
        kf = frame_camera(geometry)
        u, v = rng.uniform(0.15, 0.85) * recording.FRAME_W, rng.uniform(0.15, 0.85) * recording.FRAME_H
        target = z * np.linalg.solve(kf, [u, v, 1.0])
        t = target - r @ centre
        frames.append((render(tex, r, t, geometry), geometry, None, None))
    directory = os.path.join(tmp, "intrinsics")
    write_recording(directory, frames)
    fx, fy, cx, cy, dist, rms = calibrate_intrinsics.main(
        [directory, "--every", "1", "--fix-k3", "--out", os.path.join(tmp, "intrinsics.yaml")])
    ok = check(abs(fx / K[0, 0] - 1) < 0.005 and abs(fy / K[1, 1] - 1) < 0.005, f"fx {fx:.1f} fy {fy:.1f}")
    ok &= check(abs(cx - K[0, 2]) < 3 and abs(cy - K[1, 2]) < 3, f"cx {cx:.1f} cy {cy:.1f}")
    # 畸变按效果判：WIDE 画面范围内与无畸变相比的最大位移 / Judge the distortion by its
    # largest displacement over the WIDE view against no distortion.
    xs, ys = np.meshgrid(np.linspace(80, 1360, 33), np.linspace(24, 1048, 27))
    pts = np.stack([(xs.ravel() - cx) / fx, (ys.ravel() - cy) / fy, np.ones(xs.size)], axis=1)
    k_est = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distorted, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k_est, dist)
    plain, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k_est, None)
    shift = np.linalg.norm(distorted - plain, axis=2).max()
    ok &= check(shift < 1.0, f"distortion moves points at most {shift:.2f} px")
    return ok


def test_handeye(tmp, tex, board):
    r_mount = rot("z", 1.5) @ rot("x", -2.0) @ rot("y", 0.5)
    t_cam2body = np.array([0.02, 0.08, 0.05])
    r_cam2body = r_mount @ calibrate_handeye.OPTICAL_TO_BODY
    centre = np.concatenate(list(board.markers.values())).mean(axis=0)
    # 板竖在前方 1 m，正面朝向云台 / The board stands 1 m ahead facing the gimbal.
    r_board2world = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
    p_board = np.array([0.0, 1.0, 0.0]) - r_board2world @ centre
    frames = []
    for yaw in (-10, -5, 0, 5, 10):
        for pitch in (-6, 0, 6):
            r_bw = rot("z", yaw) @ rot("x", pitch)
            r_cam2world = r_bw @ r_cam2body
            p_cam = r_bw @ t_cam2body
            r_b2c = r_cam2world.T @ r_board2world
            t_b2c = r_cam2world.T @ (p_board - p_cam)
            gray = render(tex, r_b2c, t_b2c, WIDE)
            q = calibrate_handeye.matrix_to_quat(r_bw)
            frames.append((gray, WIDE, q, [0.5, 0, 0]))  # 转动中 / moving
            frames += [(gray, WIDE, q, [0.0, 0.0, 0.001])] * 6
    directory = os.path.join(tmp, "handeye")
    write_recording(directory, frames)
    with open(os.path.join(tmp, "true_intrinsics.yaml"), "w") as f:
        f.write(f"fx: {K[0, 0]}\nfy: {K[1, 1]}\ncx: {K[0, 2]}\ncy: {K[1, 2]}\ndistortion: [0, 0, 0, 0, 0]\n")
    q, t, rms = calibrate_handeye.main(
        [directory, "--intrinsics", os.path.join(tmp, "true_intrinsics.yaml"),
         "--out", os.path.join(tmp, "mount.yaml")])
    err = calibrate_handeye.angle_deg(calibrate_handeye.quat_to_matrix(q).T @ r_mount)
    ok = check(err < 0.05, f"mount rotation error {err:.4f} deg")
    ok &= check(np.linalg.norm(t - t_cam2body) < 0.02, f"mount translation {np.round(t, 3).tolist()}")
    return ok


def main():
    board = recording.gshang_board()
    tex = board_texture(board)
    with tempfile.TemporaryDirectory() as tmp:
        ok = test_intrinsics(tmp, tex, board)
        ok &= test_handeye(tmp, tex, board)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
