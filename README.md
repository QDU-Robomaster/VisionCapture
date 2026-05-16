# VisionCapture

VisionCapture is a sync-frame capture and calibration module. It consumes
CameraFrameSync output, records synchronized images and IMU metadata, detects
calibration boards, optionally publishes a preview overlay, and can solve camera
intrinsics from the captured GShang/ArUco calibration board views.

It is intended for capture/calibration run configs, not the normal auto-aim
runtime chain.

## Modes

- `record`: record synchronized frames and metadata.
- `calibrate_camera`: record frames and run camera intrinsic calibration once
  enough accepted board views have been collected.

`camera_calibration.enabled: true` enables the same camera calibration path even
when `mode` is left as `record`.

## Camera Calibration

The camera calibration path preserves the old CameraBase calibration behavior:

- GShang board geometry, where `square = marker * 9 / 7`;
- ArUco original dictionary marker layout;
- marker count, homography RMS, sharpness, and near-duplicate view filtering;
- OpenCV `calibrateCamera`;
- high per-view reprojection RMS outlier filtering and a final solve;
- `calibration.yml`, `views.csv`, `quality_report.txt`, debug images, and a
  pasteable `camera_info_snippet.txt`.

The output directory remains:

```text
runs/camera_calib/<timestamp>_<session>_<marker>mm_<cols>x<rows>/
```

Hand-eye calibration is not solved by this module yet. The current output is
camera intrinsics plus synchronized image/IMU data needed by a later hand-eye
solver.
