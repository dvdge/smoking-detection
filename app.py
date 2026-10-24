"""简化版：输入一个视频地址，显示疑似抽烟检测结果。"""

# 模型预处理和推理代码位于 detector.py。
from detector import OnnxDetector, ROOT
import cv2
import numpy as np
import time
import argparse
import os
from pathlib import Path

# 本地 Windows 路径在引号前加 r，避免反斜杠被当成转义符；留空时运行后输入。
VIDEO_URL = ""

# 小目标容易漏检：同时放大检测四个重叠区域。会增加推理耗时，可改 False 关闭。
SMALL_TARGET_ENHANCEMENT = True

# 当前帧短暂未检出时，保留最近结果的最长时间；黄色框表示历史结果。
RECENT_DETECTION_SECONDS = 0.6


class DetectionMemory:
    """仅保留最近检测结果，不推断遮挡原因，也不跟踪人物或烟的运动。"""
    def __init__(self, hold_seconds):
        if hold_seconds < 0:
            raise ValueError("保留时间不能小于 0。")
        self.hold_seconds = hold_seconds
        self.last_time = None
        self.last_detections = []

    def update(self, detections, timestamp):
        if detections:
            self.last_detections = [dict(item, bbox_xyxy=list(item["bbox_xyxy"]))
                                    for item in detections]
            self.last_time = timestamp
            return detections, False, 0.0
        if self.last_time is not None:
            age = timestamp - self.last_time
            if 0 <= age <= self.hold_seconds:
                # 未检出时不更新 last_time，因此历史结果一定会到期。
                return self.last_detections, True, age
        self.last_time = None
        self.last_detections = []
        return [], False, 0.0


def detect_smoking(detector, frame, enhance=True):
    """整图检测，加上四个重叠分块；把局部框换回整图坐标后去重。"""
    found = detector.predict(frame)
    if not enhance:
        return found

    height, width = frame.shape[:2]
    tile_w, tile_h = max(1, round(width * 0.6)), max(1, round(height * 0.6))
    for x in sorted({0, width - tile_w}):
        for y in sorted({0, height - tile_h}):
            crop = frame[y:y + tile_h, x:x + tile_w]
            for item in detector.predict(crop):
                detection = dict(item)
                x1, y1, x2, y2 = item["bbox_xyxy"]
                detection["bbox_xyxy"] = [x1 + x, y1 + y, x2 + x, y2 + y]
                found.append(detection)

    if not found:
        return []
    # 同一支烟可能在多个区域被识别，用 NMS 合并重复框。
    boxes = [[d["bbox_xyxy"][0], d["bbox_xyxy"][1],
              d["bbox_xyxy"][2] - d["bbox_xyxy"][0],
              d["bbox_xyxy"][3] - d["bbox_xyxy"][1]] for d in found]
    kept = cv2.dnn.NMSBoxes(boxes, [d["confidence"] for d in found],
                            detector.confidence, detector.iou)
    return [found[int(i)] for i in np.asarray(kept).reshape(-1)]


def center_window(title, frame):
    """首次显示时，将视频窗口放在 Windows 主屏幕可用区域的中间。"""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if os.name != "nt":
        return
    import ctypes
    from ctypes.wintypes import RECT

    area = RECT()
    # 0x0030 是 SPI_GETWORKAREA：读取不包含任务栏的屏幕可用区域。
    if not ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(area), 0):
        return
    screen_w, screen_h = area.right - area.left, area.bottom - area.top
    image_h, image_w = frame.shape[:2]
    scale = min(1.0, screen_w * 0.8 / image_w, screen_h * 0.8 / image_h)
    width, height = max(1, int(image_w * scale)), max(1, int(image_h * scale))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)
    x = area.left + (screen_w - width) // 2
    y = area.top + (screen_h - height) // 2
    cv2.moveWindow(title, x, y)


def detect_video(video_url, show=True, model_path=None, confidence=0.5,
                 enhance=True, hold_seconds=0.6, max_frames=0):
    """支持 RTSP、HTTP 视频直链和本地视频路径；show=False 用于离线检查。"""
    if max_frames < 0:
        raise ValueError("max_frames 不能小于 0。")
    if not video_url.strip():
        raise ValueError("视频地址不能为空。")

    # 加载模型，按设定阈值过滤低分目标。
    detector = OnnxDetector(model_path or ROOT / "models" / "smoking.onnx",
                            ["Smooking"], confidence, 0.45)

    # 打开视频。网络来源设置超时，避免地址不通时一直等待。
    is_network = video_url.lower().startswith(("rtsp://", "http://", "https://"))
    if is_network:
        cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000,
        ])
    else:
        cap = cv2.VideoCapture(video_url)

    processed = 0
    window_title = "Video smoking detection"
    memory = DetectionMemory(hold_seconds)
    try:
        if not cap.isOpened():
            raise RuntimeError("视频打不开，请检查地址、网络和密码；网页页面地址不能直接当视频地址。")
        print("视频已打开，正在检测。预览窗口按 Q 或 Esc 退出。", flush=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25.0

        while True:
            ok, frame = cap.read()             # 从视频中读取一张图片。
            if not ok:
                print("视频结束或连接中断。", flush=True)
                break

            # 本地文件按视频时间计时，避免推理较慢导致几帧就耗尽保留时间。
            timestamp = time.monotonic() if is_network else processed / fps
            detections = detect_smoking(detector, frame, enhance)
            displayed, remembered, age = memory.update(detections, timestamp)
            box_color = (0, 200, 255) if remembered else (0, 0, 255)
            for item in displayed:
                x1, y1, x2, y2 = map(int, item["bbox_xyxy"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = (f"Last seen {age:.2f}s ago" if remembered
                         else f"Smoking candidate {item['confidence']:.2f}")
                cv2.putText(frame, label,
                            (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, box_color, 2)

            if detections:
                status, color = "Suspected smoking", (0, 0, 255)
            elif remembered:
                status, color = "Recent candidate - not detected now", (0, 200, 255)
            else:
                status, color = "No smoking candidate detected", (0, 180, 0)
            cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            processed += 1

            if show:
                # 只在第一帧居中，之后仍可自由拖动窗口。
                if processed == 1:
                    center_window(window_title, frame)
                cv2.imshow(window_title, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
    return processed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="输入一个视频地址，显示疑似抽烟检测结果。")
    parser.add_argument("source", nargs="?", help="RTSP/HTTP 视频直链或本地文件路径")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "smoking.onnx")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--hold-seconds", type=float, default=RECENT_DETECTION_SECONDS)
    parser.add_argument("--no-tiles", action="store_true", help="关闭分块增强，减少计算量")
    parser.add_argument("--headless", action="store_true", help="不打开窗口")
    parser.add_argument("--max-frames", type=int, default=0, help="限制处理帧数，0 为不限")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    address = args.source or os.environ.get("VIDEO_SOURCE") or VIDEO_URL
    if not address:
        address = input("请粘贴视频地址，然后按回车：")
    try:
        count = detect_video(address.strip(), show=not args.headless,
                             model_path=args.model, confidence=args.confidence,
                             enhance=not args.no_tiles, hold_seconds=args.hold_seconds,
                             max_frames=args.max_frames)
        print(f"检测结束，共处理 {count} 帧。")
        return 0
    except KeyboardInterrupt:
        print("已停止。")
        return 0
    except ValueError as error:
        print(f"配置错误：{error}")
        return 1
    except Exception as error:
        # 避免第三方异常将含密码的视频源地址写入日志。
        print(f"运行失败（{type(error).__name__}），请检查视频地址、网络、模型与依赖。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
