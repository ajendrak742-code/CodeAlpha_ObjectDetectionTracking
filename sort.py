"""
SORT - Simple Online and Realtime Tracking.

An implementation of the algorithm from Bewley et al., "Simple Online and
Realtime Tracking" (2016), written with NumPy and SciPy.

The detector only says "there is a person at this box" for each frame on its
own. The tracker links those boxes across frames so that each object keeps
the same ID. Every frame it does three things:

  1. Predict  - a Kalman filter per object predicts where its box moves next.
  2. Match    - the Hungarian algorithm pairs predicted boxes with the new
                detections so that the total overlap (IoU) is as large as possible.
  3. Update   - matched tracks are corrected with their detection, unmatched
                detections start new tracks, and tracks that have gone
                unseen for too long are deleted.
"""

from collections import Counter, deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a, boxes_b):
    """Intersection-over-union of every box in A with every box in B.

    Boxes are [x1, y1, x2, y2]. Returns an array of shape (len(A), len(B)).
    """
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    intersection = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def box_to_measurement(box):
    """[x1, y1, x2, y2] -> [centre x, centre y, area, aspect ratio]."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return np.array([box[0] + width / 2, box[1] + height / 2, width * height, width / max(height, 1e-6)])


def state_to_box(state):
    """[centre x, centre y, area, aspect ratio, ...] -> [x1, y1, x2, y2]."""
    area, aspect = max(state[2], 1e-6), max(state[3], 1e-6)
    width = np.sqrt(area * aspect)
    height = area / width
    return np.array([state[0] - width / 2, state[1] - height / 2, state[0] + width / 2, state[1] + height / 2])


class Track:
    """One tracked object: a Kalman filter with a constant-velocity motion model.

    State: [cx, cy, area, aspect, v_cx, v_cy, v_area]. The position, size and
    shape of the box, plus how fast the position and size are changing.
    """

    # Motion model: each frame, position and area move by their velocity
    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    # Measurement model: the detector observes the first four state values
    H = np.eye(4, 7)
    # Process noise: how much the motion may deviate from constant velocity
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
    # Measurement noise: how much the detector's boxes jitter (size more than position)
    R = np.diag([1.0, 1.0, 10.0, 10.0])

    def __init__(self, box, class_id, score, trail_length=30):
        self.state = np.zeros(7)
        self.state[:4] = box_to_measurement(box)
        # Uncertainty: the box is roughly known, the velocity is not known at all yet
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

        self.id = None             # assigned once the track is confirmed
        self.hits = 1              # number of detections matched to this track
        self.frames_unseen = 0     # frames since the last matched detection
        self.score = score
        self.class_votes = Counter([class_id])
        self.trail = deque(maxlen=trail_length)  # recent positions, for drawing the path

    @property
    def box(self):
        return state_to_box(self.state)

    @property
    def class_id(self):
        """The class this object was detected as most often."""
        return self.class_votes.most_common(1)[0][0]

    def predict(self):
        """Kalman predict step: move the box forward one frame."""
        if self.state[2] + self.state[6] <= 0:
            self.state[6] = 0.0  # the area must not shrink below zero
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.frames_unseen += 1
        return self.box

    def update(self, box, class_id, score):
        """Kalman update step: correct the prediction with a matched detection."""
        residual = box_to_measurement(box) - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain: how much to trust the detection
        self.state = self.state + K @ residual
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.hits += 1
        self.frames_unseen = 0
        self.score = score
        self.class_votes[class_id] += 1


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age              # frames a track survives without a detection
        self.min_hits = min_hits            # detections needed before a track gets an ID
        self.iou_threshold = iou_threshold  # minimum overlap for a detection to match a track
        self.tracks = []
        self.total_ids = 0                  # number of IDs handed out so far

    def update(self, detections):
        """Advance the tracker by one frame.

        detections: array of rows [x1, y1, x2, y2, score, class_id].
        Returns the confirmed tracks that were detected in this frame.
        """
        # 1. Predict where every existing track is now
        predicted = np.array([track.predict() for track in self.tracks]).reshape(-1, 4)

        # 2. Match predictions with detections
        matches, unmatched_detections = self._associate(predicted, detections)

        # 3. Update matched tracks, start new ones, drop lost ones
        for track_index, detection_index in matches:
            x1, y1, x2, y2, score, class_id = detections[detection_index]
            self.tracks[track_index].update(np.array([x1, y1, x2, y2]), int(class_id), float(score))

        for detection_index in unmatched_detections:
            x1, y1, x2, y2, score, class_id = detections[detection_index]
            self.tracks.append(Track(np.array([x1, y1, x2, y2]), int(class_id), float(score)))

        self.tracks = [track for track in self.tracks if track.frames_unseen <= self.max_age]

        # A track is shown once it has been matched in min_hits frames; this
        # filters out one-frame false detections.
        visible = []
        for track in self.tracks:
            if track.frames_unseen > 0 or track.hits < self.min_hits:
                continue
            if track.id is None:
                self.total_ids += 1
                track.id = self.total_ids
            x1, y1, x2, y2 = track.box
            track.trail.append((int((x1 + x2) / 2), int(y2)))  # bottom centre of the box
            visible.append(track)
        return visible

    def _associate(self, predicted, detections):
        """Pair tracks with detections. Returns (matches, unmatched detection indices)."""
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        overlap = iou_matrix(predicted, detections[:, :4])
        # Hungarian algorithm: the one-to-one pairing with the highest total IoU
        track_indices, detection_indices = linear_sum_assignment(-overlap)

        matches = [(t, d) for t, d in zip(track_indices, detection_indices)
                   if overlap[t, d] >= self.iou_threshold]
        matched_detections = {d for _, d in matches}
        unmatched = [d for d in range(len(detections)) if d not in matched_detections]
        return matches, unmatched
