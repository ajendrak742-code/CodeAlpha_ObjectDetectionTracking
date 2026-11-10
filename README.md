# Object Detection and Tracking

Detects objects in a live webcam feed or a video file and follows each one across frames, drawing a bounding box, a class label and a persistent tracking ID.

Built for **Task 4** of the CodeAlpha Artificial Intelligence internship.

- **Detection:** YOLOX, a pre-trained YOLO model (80 COCO classes), run with OpenCV's DNN module
- **Tracking:** SORT (Kalman filter + Hungarian algorithm), implemented from the paper in `sort.py`
- **Runs on a CPU** in real time. No GPU and no deep learning framework required.

## Features

- Webcam or video file input
- Bounding box, label and tracking ID for every object, for example `person #3`
- A coloured trail showing the path each object has taken
- Live counter: frames per second, objects being tracked, total objects seen
- Optional: track only chosen classes, save the annotated video, choose a larger model
- Summary at the end: how many objects of each class were tracked

## Run it

Requires Python 3.10 or newer.

```bash
git clone https://github.com/ajendrak742-code/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

```bash
python detect_track.py                                 # webcam
python detect_track.py --source path/to/video.mp4      # video file
```

The model file (20 MB) is downloaded automatically the first time. Press `Q` or `Esc` to quit.

To try it on a sample video of pedestrians from the OpenCV project:

```bash
curl -L -o samples/vtest.avi https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi
python detect_track.py --source samples/vtest.avi
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--source` | `0` | Webcam number, or path to a video file |
| `--model` | `yolox_tiny` | `yolox_nano`, `yolox_tiny`, `yolox_s` or `yolox_m`. Larger models are more accurate and slower |
| `--conf` | `0.5` | Minimum detection confidence |
| `--classes` | all | Only track these classes, e.g. `--classes person car` |
| `--save FILE` | off | Also write the annotated video to a file, e.g. `--save output.mp4` |
| `--no-show` | off | Do not open a window (use with `--save`) |
| `--no-trails` | off | Do not draw the path behind each object |
| `--max-age` | `20` | Frames a lost object keeps its ID before the track is deleted |
| `--min-hits` | `3` | Detections needed before an object gets an ID |

## How it works

For every frame:

```
frame ──► YOLOX detector ──► boxes + class + score
                                   │
                                   ▼
                             SORT tracker
              1. Predict   Kalman filter moves each known object's box forward
              2. Match     Hungarian algorithm pairs predictions with new boxes (by IoU)
              3. Update    matched tracks corrected · new tracks started · lost tracks removed
                                   │
                                   ▼
                  draw box, label and ID  ──► display / save
```

**Detection (`detector.py`).** The frame is resized to the model's input size with letterboxing and
passed through YOLOX. The raw output is decoded from grid coordinates to pixel boxes, low-confidence
boxes are dropped and non-maximum suppression removes duplicates.

**Tracking (`sort.py`).** A detector looks at each frame on its own, so it cannot tell whether the
person in this frame is the same one as in the last frame. SORT solves this:

- Each tracked object has a **Kalman filter** with a constant-velocity model. Its state is the box
  centre, area and aspect ratio plus their velocities. Before looking at the new frame, the filter
  predicts where the box should be now.
- The **Hungarian algorithm** finds the one-to-one pairing between predicted boxes and new
  detections that gives the highest total overlap (IoU). Pairs that overlap too little are rejected.
- Matched tracks are corrected with their detection. An unmatched detection starts a new track,
  which gets an ID after it has been seen in 3 frames (this filters out one-frame false detections).
  A track that goes unseen for 20 frames is deleted.

## Results

On the OpenCV sample video (768×576, 795 frames) with the default `yolox_tiny` model, the pipeline
runs at about 14 frames per second on a laptop CPU, including drawing and saving the video.

## Limitations

- SORT matches objects only by position and motion. After a long occlusion, or when two people
  cross closely, an object can come back with a new ID. Deep SORT reduces this by also comparing
  appearance.
- The tiny model occasionally mislabels small or distant objects. `--model yolox_s` is more accurate
  but about three times slower on a CPU; raising `--conf` also reduces false detections.

## Project structure

```
detect_track.py    Main program: video input, drawing, display, saving
detector.py        YOLOX detector on OpenCV DNN (preprocessing, decoding, NMS)
sort.py            SORT tracker (Kalman filter, IoU matching, track management)
requirements.txt   Python dependencies
models/            Model file, downloaded on first run (not in the repository)
samples/           Put sample videos here (not in the repository)
```

## References

- Z. Ge et al., "YOLOX: Exceeding YOLO Series in 2021". Pre-trained models from the
  [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0 licence).
- A. Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.

## Tech stack

Python · OpenCV · NumPy · SciPy
