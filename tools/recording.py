"""统一录像格式的读取与 GShang 标定板检测，供离线标定工具共用。

Reading the unified recording format and detecting the GShang board, shared by the
offline calibration tools.

录像目录 / A recording directory:
  frames.csv  frame,timestamp_us,frame_counter,roi_x,roi_y,decimation[,qw..az]
  NNNNNN.pgm  640x512 BayerRG8 (R at 0,0)
"""
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 512


@dataclass
class Frame:
    frame: int
    timestamp_us: int
    roi_x: int
    roi_y: int
    decimation: int
    quat_wxyz: np.ndarray | None  # 本体系到世界系 / body to world
    gyro: np.ndarray | None       # 本体系 rad/s / body rad/s


def load_frames(directory):
    """读 frames.csv；没有 IMU 列时 quat_wxyz 与 gyro 为 None。"""
    rows = []
    with open(os.path.join(directory, "frames.csv"), newline="") as f:
        for r in csv.DictReader(f):
            imu = "qw" in r
            rows.append(Frame(
                int(r["frame"]), int(r["timestamp_us"]), int(r["roi_x"]), int(r["roi_y"]),
                int(r["decimation"]),
                np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")]) if imu else None,
                np.array([float(r[k]) for k in ("gx", "gy", "gz")]) if imu else None))
    return rows


def read_gray(directory, frame):
    """读一帧 PGM 并转灰度（RGGB 在 OpenCV 里叫 BayerBG）。"""
    raw = cv2.imread(os.path.join(directory, f"{frame:06d}.pgm"), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape != (FRAME_H, FRAME_W):
        raise ValueError(f"{directory}/{frame:06d}.pgm is not a 640x512 PGM")
    return cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)


def frame_to_native(points, row):
    """帧像素到原生像素；跳采 2 的采样相位为 -0.5（CameraTypes::FrameToNative）。"""
    phase = -0.5 if row.decimation == 2 else 0.0
    p = np.asarray(points, np.float64).reshape(-1, 2)
    return np.stack([row.roi_x + phase + row.decimation * p[:, 0],
                     row.roi_y + phase + row.decimation * p[:, 1]], axis=1)


@dataclass
class Board:
    markers: dict      # id -> 4 角（m），左上、右上、右下、左下 / corners TL, TR, BR, BL
    square_m: float    # 棋盘格边长 / chessboard square side
    cols: int
    rows: int
    cell_ids: dict     # (行, 列) -> marker id / (row, col) -> marker id


def gshang_board(marker_mm=25.0, cols=8, rows=6, marker_cells=7, square_cells=9):
    """GShang 板（ChArUco 式）：cols x rows 个棋盘格，(行 + 列) 为偶数的格是黑格，其余白格中
    各一个 ArUco（DICT_ARUCO_ORIGINAL，按行优先编号）。格边长 = marker 边长 / 7 × 9，
    marker 距格左上一个码元。

    GShang board (ChArUco style): cols x rows squares; squares with an even row + column
    are black, every other one holds an ArUco (DICT_ARUCO_ORIGINAL, numbered row-major).
    The square is the marker side / 7 x 9 and the marker sits one code cell inside.
    """
    cell = marker_mm / marker_cells
    square = cell * square_cells
    markers, cell_ids, marker_id = {}, {}, 0
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                continue
            x0, y0 = c * square + cell, r * square + cell
            x1, y1 = x0 + marker_mm, y0 + marker_mm
            markers[marker_id] = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]],
                                          np.float64) * 1e-3
            cell_ids[(r, c)] = marker_id
            marker_id += 1
    return Board(markers, square * 1e-3, cols, rows, cell_ids)


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL), params)


def detect_board(detector, gray, board):
    """返回棋盘内角点 (物体点 Nx3, 帧像素 Nx2)；一个也没有时为 None。

    每个内角点挨着两个 marker 格：两个 marker 都检测到时，用它们 8 个角点的局部单应预测
    内角点，再用 cornerSubPix 精化。用棋盘的鞍点而不用 marker 角点：marker 角点是模糊
    黑方块的直角，检测结果整体向内偏约 0.3–0.5 px，且偏差随编码不同。
    Every inner corner touches two marker squares: when both markers are found, the
    local homography of their eight corners predicts the inner corner and cornerSubPix
    refines it. The chessboard saddle points are used instead of marker corners, which
    are the right-angle corners of a blurred black square and sit 0.3–0.5 px inside,
    by an amount that varies with the code.
    """
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    found = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())
             if int(i) in board.markers}
    obj, pred, sides = [], [], []
    for r in range(1, board.rows):
        for c in range(1, board.cols):
            near = [board.cell_ids[k] for k in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c))
                    if k in board.cell_ids]
            if len(near) != 2 or not all(m in found for m in near):
                continue
            src = np.concatenate([board.markers[m][:, :2] for m in near])
            dst = np.concatenate([found[m] for m in near])
            h, _ = cv2.findHomography(src, dst)
            if h is None:
                continue
            p = h @ np.array([c * board.square_m, r * board.square_m, 1.0])
            obj.append([c * board.square_m, r * board.square_m, 0.0])
            pred.append(p[:2] / p[2])
            sides.append(min(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[5] - dst[4])))
    if not obj:
        return None
    # 窗口取 marker 边长的 1/5，留在两个黑格之内 / Window of 1/5 marker side, inside the
    # two black squares.
    half = max(2, int(min(sides) / 5))
    img = cv2.cornerSubPix(gray, np.array(pred, np.float32).reshape(-1, 1, 2), (half, half),
                           (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    return np.array(obj), img.reshape(-1, 2).astype(np.float64)
