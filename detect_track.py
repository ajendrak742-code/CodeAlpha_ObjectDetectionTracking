"""
Object Detection and Tracking
CodeAlpha Artificial Intelligence Internship - Task 4

Reads video from a webcam or a file, detects objects in every frame with a
pre-trained YOLOX model, follows them across frames with the SORT tracker
and shows the result live: a box, a label and a tracking ID for each object.

Examples:
  python detect_track.py                               webcam
  python detect_track.py --source samples/vtest.avi    video file
  python detect_track.py --source samples/vtest.avi --save output.mp4
  python detect_track.py --classes person car          only track people and cars

Press Q or Esc to quit.
"""

import argparse
import sys
import time
from collections import Counter

import cv2

from detector import COCO_CLASSES, INPUT_SIZE, YoloxDetector
from sort import Sort

WINDOW_TITLE = "Object Detection and Tracking  (press Q to quit)"

# Distinct box colours (BGR); a track keeps one colour for its whole life
COLOURS = [
    (66, 135, 245), (60, 180, 75), (48, 130, 245), (200, 130, 0), (180, 30, 145),
    (240, 240, 70), (230, 50, 240), (60, 245, 210), (190, 190, 250), (128, 128, 0),
    (255, 190, 220), (40, 110, 170), (200, 250, 255), (0, 0, 128), (195, 255, 170),
    (0, 128, 128), (180, 215, 255), (128, 0, 0), (75, 25, 230), (25, 225, 255),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking (YOLOX + SORT).")
    parser.add_argument("--source", default="0", help="webcam number (default 0) or path to a video file")
    parser.add_argument("--model", default="yolox_tiny", choices=sorted(INPUT_SIZE),
                        help="detector size: tiny is fastest, s and m are more accurate but slower")
    parser.add_argument("--conf", type=float, default=0.5, help="minimum detection confidence (0 to 1)")
    parser.add_argument("--classes", nargs="+", metavar="NAME",
                        help="only track these classes, e.g. --classes person car")
    parser.add_argument("--save", metavar="FILE", help="also write the annotated video to this file (.mp4)")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (use with --save)")
    parser.add_argument("--no-trails", action="store_true", help="do not draw the path behind each object")
    parser.add_argument("--max-age", type=int, default=20, help="frames a lost object keeps its ID")
    parser.add_argument("--min-hits", type=int, default=3, help="detections needed before an object gets an ID")
    args = parser.parse_args()

    if args.classes:
        unknown = [name for name in args.classes if name not in COCO_CLASSES]
        if unknown:
            parser.error(f"unknown class {unknown[0]!r}. Choose from: {', '.join(COCO_CLASSES)}")
    return args


def draw_track(frame, track, show_trail):
    colour = COLOURS[track.id % len(COLOURS)]
    x1, y1, x2, y2 = (int(v) for v in track.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    if show_trail and len(track.trail) > 1:
        for start, end in zip(list(track.trail)[:-1], list(track.trail)[1:]):
            cv2.line(frame, start, end, colour, 2)

    # Label on a filled background so it stays readable: "person #3"
    label = f"{COCO_CLASSES[track.class_id]} #{track.id}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 6, top + text_h + baseline + 4), colour, -1)
    brightness = 0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2]
    text_colour = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(frame, label, (x1 + 3, top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colour, 1, cv2.LINE_AA)


def draw_status(frame, fps, visible, total):
    text = f"FPS {fps:4.1f}   Tracking {visible}   Total seen {total}"
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (0, 0), (text_w + 20, text_h + baseline + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, text_h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()

    # A number means a webcam, anything else is a path to a video file
    is_webcam = args.source.isdigit()
    capture = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not capture.isOpened():
        sys.exit(f"Could not open video source: {args.source}")

    detector = YoloxDetector(args.model, conf_threshold=args.conf)
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits)
    wanted = {COCO_CLASSES.index(name) for name in args.classes} if args.classes else None

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not 1 <= source_fps <= 120:
        source_fps = 30.0  # webcams often do not report a frame rate
    writer = None
    fps = 0.0
    frames = 0
    seen = {}  # track ID -> class name, for the summary at the end
    started = time.perf_counter()

    while True:
        frame_start = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break  # end of the video, or the camera was disconnected

        # Detect -> (optional class filter) -> track
        detections = detector.detect(frame)
        if wanted is not None:
            detections = detections[[int(class_id) in wanted for class_id in detections[:, 5]]]
        tracks = tracker.update(detections)

        for track in tracks:
            draw_track(frame, track, show_trail=not args.no_trails)
            seen[track.id] = COCO_CLASSES[track.class_id]
        draw_status(frame, fps, len(tracks), tracker.total_ids)
        frames += 1

        if args.save:
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
                if not writer.isOpened():
                    sys.exit(f"Could not write to {args.save}")
            writer.write(frame)

        elapsed = time.perf_counter() - frame_start
        if not args.no_show:
            cv2.imshow(WINDOW_TITLE, frame)
            # For a video file, wait so that it is not played faster than it was recorded
            wait_ms = 1 if is_webcam else max(1, int((1 / source_fps - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            elapsed = time.perf_counter() - frame_start

        # Smoothed frames-per-second for the on-screen counter
        fps = 1 / elapsed if fps == 0 else 0.9 * fps + 0.1 / elapsed

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    total_time = time.perf_counter() - started
    print(f"Processed {frames} frames in {total_time:.1f} s ({frames / max(total_time, 1e-6):.1f} FPS).")
    print(f"Objects tracked: {tracker.total_ids}")
    for name, count in Counter(seen.values()).most_common():
        print(f"  {name}: {count}")
    if args.save and writer is not None:
        print(f"Saved annotated video to {args.save}")


if __name__ == "__main__":
    main()
