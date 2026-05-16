# VisionCapture

VisionCapture is a sync-frame capture and calibration data module. It consumes
CameraFrameSync output, records synchronized images and IMU metadata, detects
calibration boards, and optionally publishes a preview overlay.

It is intended for capture/calibration run configs, not the normal auto-aim
runtime chain. The calibration result should still be written back to the
tracker hand-eye configuration.
