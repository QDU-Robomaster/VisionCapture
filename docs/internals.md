# VisionCapture 内部机制 / Internals

本页记录 VisionCapture 的实现细节，用于维护模块和排查采集问题。使用模块时依赖的约定见 [README](../README.md)。

This page records the implementation details of VisionCapture, for maintaining the Module and investigating capture problems. The conventions that users of the Module rely on are in the [README](../README.md).

## 1. 帧队列与工作线程 / Frame Queue and Worker Thread

同步 Topic 回调只复制 `SharedFrame` 句柄，并写入容量为 2、满时丢弃最旧一帧的队列。OpenCV、预览深拷贝和磁盘 I/O 都在模块拥有的工作线程中执行。RGB8 / RGBA8 输入在工作线程内转换为 OpenCV 的 BGR / BGRA 约定后再检测、预览和写图。工作线程处理异常时关闭队列、释放待处理帧并锁存会话失败。

The synchronized Topic callback only copies the `SharedFrame` handle and writes it into a drop-oldest queue of capacity 2. OpenCV, the preview deep copy and the disk I/O all run on the worker thread owned by the Module. RGB8 / RGBA8 input is converted in the worker to the OpenCV BGR / BGRA convention before detection, preview and image writing. When the worker raises an exception, the queue is closed, the pending frames are released and the session failure is latched.

## 2. 析构顺序 / Destruction Order

析构时先停止可取消的 stdin 控制读取器并等待其退出，再停止帧工作线程并等待其退出。

The destructor first stops the cancellable stdin control reader and waits for it to exit, then stops the frame worker thread and waits for it to exit.

## 3. 会话失败与求解提交 / Session Failure and Solver Submission

任一必需记录或 CSV 刷盘失败时，会话失败被锁存，内参求解视角被清空，`solve` 不再返回 PASS。图像和元数据完整持久化后，该帧才提交给内参求解器。

When any required record or CSV flush fails, the session failure is latched, the intrinsic solver views are cleared and `solve` no longer returns PASS. A frame is submitted to the intrinsic solver after its image and metadata are completely persisted.

## 4. 求解与写盘 / Solve and Write

质量检查通过后，所有输出逐字节写后读回成功，求解才返回成功并生成 `calibration.yml` 与 `camera_info_snippet.txt`。自动保存与 stdin `solve` 共用同一次求解与写盘，并发的重复请求等待并复用同一结果。

After the quality check passes, the solve returns success and generates `calibration.yml` and `camera_info_snippet.txt` only when every output has been written and read back byte for byte. The automatic save and the stdin `solve` share one solve-and-write run, and concurrent duplicate requests wait for and reuse the same result.
