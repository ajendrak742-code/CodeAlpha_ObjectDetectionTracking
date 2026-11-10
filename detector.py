"""
Object detector: a pre-trained YOLOX model run with OpenCV's DNN module.

YOLOX is an anchor-free member of the YOLO family, trained on the COCO
dataset (80 everyday classes such as person, car, dog, chair). The model is
stored in ONNX format, so OpenCV can run it directly on the CPU and no deep
learning framework has to be installed.
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"
MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.onnx"
# Side length of the square image each model expects
INPUT_SIZE = {"yolox_nano": 416, "yolox_tiny": 416, "yolox_s": 640, "yolox_m": 640}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_model_path(name):
    """Return the path of the ONNX file, downloading it the first time it is needed."""
    path = MODEL_DIR / f"{name}.onnx"
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"Downloading {name}.onnx (first run only)...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL.format(name=name), partial)
        partial.rename(path)
    return path


class YoloxDetector:
    def __init__(self, model="yolox_tiny", conf_threshold=0.5, nms_threshold=0.45):
        self.net = cv2.dnn.readNetFromONNX(str(get_model_path(model)))
        self.size = INPUT_SIZE[model]
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.grid, self.stride = self._build_grid()

    def _build_grid(self):
        """YOLOX predicts boxes on three grids (cells of 8, 16 and 32 pixels).

        Returns, for every prediction, the (x, y) position of its grid cell
        and the cell size, which are needed to turn the raw output into pixels.
        """
        grids, strides = [], []
        for stride in (8, 16, 32):
            cells = self.size // stride
            ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
            grids.append(np.stack((xs, ys), axis=-1).reshape(-1, 2))
            strides.append(np.full((cells * cells, 1), stride))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def detect(self, frame):
        """Find objects in a BGR frame.

        Returns an array with one row per object: [x1, y1, x2, y2, score, class_id].
        """
        # 1. Resize the frame to fit the model's square input, keeping its shape
        #    and filling the leftover area with grey ("letterboxing").
        height, width = frame.shape[:2]
        ratio = min(self.size / height, self.size / width)
        resized = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized

        # 2. Run the network. Output: one row per candidate box,
        #    [centre x, centre y, width, height, objectness, 80 class scores].
        self.net.setInput(cv2.dnn.blobFromImage(canvas))
        output = self.net.forward()[0]

        # 3. Keep only confident candidates. score = objectness x best class score
        class_scores = output[:, 4:5] * output[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        keep = scores >= self.conf_threshold
        if not keep.any():
            return np.empty((0, 6), dtype=np.float32)
        output, class_ids, scores = output[keep], class_ids[keep], scores[keep]
        grid, stride = self.grid[keep], self.stride[keep]

        # 4. Decode the boxes: grid units -> pixels in the original frame
        centre = (output[:, :2] + grid) * stride / ratio
        size = np.exp(output[:, 2:4]) * stride / ratio
        top_left = centre - size / 2

        # 5. Non-maximum suppression removes duplicate boxes on the same object
        boxes_xywh = np.hstack((top_left, size))
        chosen = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)
        chosen = np.array(chosen, dtype=int).reshape(-1)

        detections = np.column_stack((top_left, top_left + size, scores, class_ids))[chosen]
        detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, width - 1)
        detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, height - 1)
        return detections.astype(np.float32)
