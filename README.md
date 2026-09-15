# Vehicle Tracker

Real-time traffic analysis from an RGB camera stream. The program takes an RTSP URL, finds
vehicles, gives each one a stable ID, reads its licence plate, estimates its colour and draws
a box in that colour with the plate next to it.

![Dashcam clip: both plates read, boxes coloured by vehicle](docs/dashcam.jpg)

![Dense traffic clip: eight vehicles tracked, plate KE15NCC read](docs/dense_traffic.jpg)

The second frame also shows the weakest part: the silver cars (IDs 52 and 42) are labelled
`white`. See [Limitations](#limitations).

## Task coverage

| Requirement | How it is done |
|---|---|
| Continuous analysis of a camera stream | RTSP read in a background thread that keeps the newest frame; timeouts and reconnects |
| New vehicles and a unique ID for each | YOLO11n detector with ByteTrack; a new track ID is a new vehicle |
| Licence plates | plate detector and OCR on the vehicle crop, voted on across the track |
| Vehicle colour | HSV rule on the body panel, voted on across the track |
| Real time and edge inference | per-vehicle cache (2.6–3.9× faster), ONNX Runtime thread fixes, measured on 4 CPU cores; OpenVINO and INT8 evaluated |
| Takes an RTSP URL, shows video, box in the vehicle colour, plate | `python -m vehicle_tracker --source rtsp://...` |
| Optional: skip frames without vehicles or readable plates | analysis skips empty frames; `--filter-stream` hides frames without a known plate |

In short: pretrained models where they already solve the problem (vehicles are COCO classes,
plate models exist for this exact task), and the effort spent on what decides speed and quality
on a weak device: computing colour and plate once per vehicle instead of once per frame, voting
out misreadings, and measuring every choice on video. No model was trained, because the task
comes with no labelled data from its camera. The first place where training would pay off is
colour, see [Limitations](#limitations).

## Contents

- [Task coverage](#task-coverage)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Results](#results)
- [Optimisation experiments that were not adopted](#optimisation-experiments-that-were-not-adopted)
- [Running on an edge device](#running-on-an-edge-device)
- [Limitations](#limitations)
- [Development](#development)
- [Licence](#licence)

## Quick start

Requires Python 3.12 or 3.13. Tested on Windows 11 with Python 3.13.5; the Docker image is
built on Linux in CI.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

# optional, NVIDIA GPU: install the CUDA build of torch first
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt

# YOLO11n weights, both plate models and the two sample videos (54 MB)
python scripts/fetch_assets.py
```

Run on a camera:

```bash
python -m vehicle_tracker --source rtsp://user:password@192.168.1.10:554/stream
```

Or on a sample video:

```bash
python -m vehicle_tracker --source data/dashcam_highway.mp4
python -m vehicle_tracker --source data/highway_traffic.mp4
```

Press `q` or `Esc` in the window to stop. A summary with FPS and per-stage timings is
printed on exit.

| Option | Default | |
|---|---|---|
| `--source` | required | RTSP/RTMP/HTTP URL, video file or webcam index (`0`) |
| `--device` | `auto` | `cpu`, `cuda` or `auto` (CUDA if torch can see a GPU) |
| `--conf` | `0.5` | vehicle detection confidence |
| `--imgsz` | `640` | detector input size |
| `--weights` | `weights/yolo11n.pt` | detector weights |
| `--no-display` | off | no window, for servers and containers |
| `--save-video PATH` | off | write the annotated video to a file |
| `--filter-stream` | off | show and save only frames with a vehicle whose plate is known |

The log gets one line per identified vehicle:

```text
00:41:51  INFO    vehicle 8: grey, plate CF5775
00:42:05  INFO    vehicle 10: grey, plate NU04703
```

### Testing RTSP without a camera

Any video can be served as a camera with [MediaMTX](https://github.com/bluenviron/mediamtx/releases)
and ffmpeg. Start `mediamtx` in one terminal, then in another:

```bash
ffmpeg -re -stream_loop -1 -i data/dashcam_highway.mp4 -an -c:v libx264 -preset ultrafast \
       -tune zerolatency -g 30 -f rtsp rtsp://127.0.0.1:8554/cam
```

and run the tracker with `--source rtsp://127.0.0.1:8554/cam`. Stopping ffmpeg shows the
reconnect logic: five attempts with a growing delay, then one error line and exit code 1.

An Android phone with the IP Webcam app is a quick way to test with a real camera:
`--source rtsp://<phone-ip>:8080/h264_pcm.sdp`.

### Docker

A CPU image with the models baked in, so it starts on a device that is offline:

```bash
docker build -t vehicle-tracker .
docker run --rm -v "$PWD/data:/data" vehicle-tracker \
    --source rtsp://192.168.1.10:554/stream --no-display --save-video /data/annotated.mp4
```

The image has no display, so `--no-display` is required. The container stops on SIGINT, so
`docker stop` goes through the same shutdown as Ctrl+C and closes the output file.

### Troubleshooting

- **The window does not open.** The plate model packages depend on `opencv-python-headless`,
  which can overwrite the windowed OpenCV. The program detects this at start and says so. Fix:
  `pip install --force-reinstall opencv-python`.
- **`no CUDA execution provider available, plate models run on the CPU`** is expected with the
  default `onnxruntime` package: the vehicle detector runs on the GPU, the two small plate models
  on the CPU. Install `onnxruntime-gpu` to move them to the GPU as well.

## How it works

Each frame goes through these stages:

1. **Reader thread** takes the newest frame from the RTSP stream.
2. **YOLO11n + ByteTrack** find vehicles and give them IDs, on every frame.
3. **Track registry** keeps state per vehicle and decides what still needs computing.
4. **Colour** is estimated on a few frames per vehicle and voted on.
5. **Plate detector and OCR** run on a schedule, at most two vehicles per frame, and the
   readings are voted on per vehicle.
6. **Overlay and log**: the box in the vehicle colour, its ID and plate, and one log line per
   vehicle.

The sections below explain each stage and why it is built this way.

### 1. Video input

Frames are read in a background thread. For a stream only the newest frame is kept, so when
processing is slower than the camera the program skips frames instead of falling behind and
building up latency. For a file every frame is kept, so runs and benchmarks are reproducible.

RTSP is opened through FFmpeg with explicit 5 s open and read timeouts. Without them a dropped
stream took about 8 s to notice and a stalled connection could hang much longer. A lost stream
is retried five times with exponential backoff; if the camera does not come back the program
prints one error line, still writes the run summary, closes the output file and exits with
code 1.

### 2. Detection and unique IDs

[YOLO11n](https://docs.ultralytics.com/models/yolo11/) (2.6M parameters, the smallest model of
the family) detects the COCO classes car, motorcycle, bus and truck. It was chosen because
vehicles need no custom training, and because the same weights export to TensorRT, OpenVINO and
ONNX, which covers the usual edge targets. On CUDA it runs in FP16.

IDs come from ByteTrack. Unlike DeepSORT or BoT-SORT with re-identification, it matches boxes by
motion and overlap only, with no second network per vehicle, so tracking adds almost nothing to
the frame budget. The price is that a vehicle hidden for too long gets a new ID.

The confidence threshold is 0.5. Lower values let distant vehicles and roadside objects flicker
in and out, getting a new ID every time: at 0.35 one clip produced 75 IDs over 420 frames instead
of 18, and at 0.45 the dense clip gives 102 tracks instead of 49 and a false plate appears on the
dashcam clip.

A new vehicle is simply a track ID seen for the first time; `tracks seen` in the summary counts
them.

### 3. Per-vehicle cache: the main optimisation

Colour and plate belong to the vehicle, not to the frame, so they are not recomputed on every
frame. The track registry keeps state per ID and decides what is worth computing:

- **Frames without vehicles** stop after detection.
- **Colour** is sampled every 5th frame, at most 15 times per vehicle, and decided by majority
  vote, so one blurred frame cannot change the answer.
- **Plates** are read every 5th frame, only for vehicles taller than 8% of the frame, at most two
  plate reads per frame, longest-waiting vehicle first. Vehicles entering together get staggered
  schedules so they do not all land on the same frame.
- **Reading stops** once the plate is settled (see below), or after 40 attempts.
- **The budget is renewed** when a vehicle has doubled in height. A bus in the dashcam clip is in
  view for 377 frames, but its plate is legible only in the last ~40 of them; the first budget
  was spent while it was far away. Doubling the height roughly doubles the plate width, which is
  the difference between 13% and 82% correct characters (see step 4). A factor of 1.5 wasted part
  of the new budget too early; 3.0 was never reached.

The cache makes the pipeline 2.6–3.9× faster with the same plates, and on the dense clip it is
also more accurate, see [Results](#results).

### 4. Licence plates

Two small ONNX models from the same author, both MIT-licensed:

- plate detector [`open-image-models`](https://github.com/ankandrew/open-image-models)
  `yolo-v9-t-384`, run on the vehicle crop;
- OCR [`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr) `cct-xs-v2-global`.

They were chosen over general-purpose OCR such as EasyOCR or PaddleOCR because they are trained
for plates, small, and run on ONNX Runtime without the GPU: a full plate read (detector and OCR)
takes about 15 ms on the desktop CPU.

Choices, each measured on the sample footage:

- **Input 384** finds a plate on 92% of vehicle crops, against 84% at 256 and 72% at 640:
  upscaling a small crop adds no detail.
- **The extra-small OCR model** was as accurate as the larger `cct-s` at a sixth of the cost.
- **Plates narrower than 40 px are skipped.** From 40 px up 82% of characters are correct, at
  20–39 px only 13%, and below that the recognizer returns nonsense with 0.98 confidence, so
  confidence alone cannot filter it.
- **A reading must contain letters and digits.** This drops text read off badges and dealer
  frames. It also rejects vanity plates without digits (2.8% of the OCR benchmark).

A single frame is right only about a third of the time, so readings are voted on per track.
Each vote is weighted by the OCR confidence, and a reading in the Ukrainian format (`AA1234BB`)
weighs 1.5× so that it wins ties, while foreign plates are still read. A plate is reported only
when all of these hold:

| Rule | Value | Why |
|---|---|---|
| agreeing readings | ≥ 2 | one crisp local-format reading already reaches the score |
| vote score | ≥ 1.5 | at 2.0 the one legible plate of the dense clip (KE15NCC, 1.87) was lost |
| lead over the runner-up | ≥ 0.9 | rival readings differ by one character; wrong leaders led by ≤ 0.66, correct ones by ≥ 0.92 |
| vehicle has moved | ≥ 10% of its height | a roadside sign detected as a bus has a crisp, never-moving ID number; false detections moved ≤ 3%, real vehicles ≥ 27% |

The plate appears on screen as soon as it is reported. The log line is written only when the
plate is settled (score ≥ 5 with the same lead, after which it is no longer read and cannot
change), or when the vehicle leaves before that. Without this the CPU run first logged CF5715 and
corrected it to CF5775 a second later.

### 5. Colour

A rule on the HSV values of the body panel: the middle 50% of the box width and 40–85% of its
height, which avoids most of the windscreen (it reflects the sky) and the road below.

- median brightness below 110 → `black` (dark paint has noisy hue and came out blue at 70);
- low saturation → `grey` or `white` by brightness;
- otherwise the dominant hue from a smoothed circular histogram → `red`, `orange`, `yellow`,
  `green` or `blue`.

It costs under 0.1 ms per sample and needs no training data. The box and label are drawn in the
detected colour. Accuracy is 70% of vehicles; the reasons are in [Limitations](#limitations).

### 6. Input stream filter (optional task)

Frames without vehicles never reach the analysis stages. With `--filter-stream` the program also
stops showing and saving frames that do not contain a vehicle with a known plate. On the dashcam
clip it keeps 662 of 825 frames.

### 7. Threads: the largest single speed-up

Two problems made the CPU pipeline several times slower than its parts:

- Every ONNX Runtime session opens a thread pool the size of the machine. Two plate models plus
  the detector on 24 threads competed for the same cores, and the detector lost: 80 ms per frame
  inside the pipeline against 18.7 ms on its own. The plate models are now limited to 4 threads
  (fewer on small machines).
- ONNX Runtime threads busy-wait after each call. The plate models run only on some frames, so
  the spinning mostly took cores from the detector. It is disabled.

| Change | Machine | Before | After |
|---|---|---|---|
| limit plate model threads | 24 threads, frames preloaded | 11.5 fps | 40.8 fps |
| disable thread spinning | 4 cores (see below) | 5.2 fps | 13.9 fps |

Both changes left tracks and plates identical.

## Results

### Speed

Intel i7-13700KF, RTX 4070 Ti, Windows 11, Python 3.13.5, torch 2.13.0+cu130,
onnxruntime 1.30.0 (CPU build, so plate models always run on the CPU). Measured with
`scripts/benchmark.py`: 400 frames after 20 warm-up frames, each configuration in a separate
process. FPS is processing throughput: decoding runs in the reader thread and the display is off.
Repeated runs differ by up to ~7%.

"Uncached" switches off the per-vehicle cache: colour and plate are computed for every vehicle on
every frame.

Dashcam clip, 1920×1080, 30 fps, few vehicles, large plates:

| Device | Cache | FPS | Detect, ms | Plate, ms | OCR calls per frame | Plates |
|---|---|---|---|---|---|---|
| CPU | on | **45.3** | 18.2 | 3.7 | 0.25 | CF5775, NU04703 |
| CPU | off | 16.9 | 18.0 | 40.7 | 2.81 | CF5775, NU04703 |
| CUDA | on | **80.2** | 9.2 | 3.1 | 0.24 | CF5775, NU04703 |
| CUDA | off | 20.5 | 13.8 | 34.5 | 2.81 | CF5775, NU04703 |

Dense traffic clip, 1920×1080, 25 fps, many vehicles, small plates:

| Device | Cache | FPS | Detect, ms | Plate, ms | OCR calls per frame |
|---|---|---|---|---|---|
| CPU | on | **34.5** | 18.1 | 10.7 | 0.70 |
| CPU | off | 13.5 | 18.2 | 55.5 | 3.66 |
| CUDA | on | **52.7** | 9.7 | 9.1 | 0.70 |
| CUDA | off | 15.6 | 16.0 | 47.5 | 3.65 |

The cache gives 2.7× on CPU and 3.9× on CUDA on the dashcam clip, 2.6× and 3.4× on dense
traffic. On the full dense clip it is also more accurate: with the cache the program reports
only KE15NCC, without it the CPU run reports `186K13, N411MM, E001YE, KE15NCC`, three of them
invented. More attempts on unreadable plates give a stable misreading more chances to collect
votes.

### A small CPU as an edge approximation

No edge board was available, so the whole pipeline was pinned to four efficiency cores of the
same i7 (no GPU). These cores are likely faster than the Cortex-A78AE cores of a Jetson Orin, so
treat this as an upper bound, not a promise.

| Clip | Source FPS | Processed FPS |
|---|---|---|
| dashcam | 30 | 14.7 |
| dense traffic | 25 | 11.6 |

On four cores the program processes about every second frame of a stream; the reader drops the
rest, so latency does not grow. Most of the frame time (50 of about 66 ms at 640) is the detector
network itself, not overhead, so the next step is an inference runtime built for the target
device, see [Running on an edge device](#running-on-an-edge-device). Memory stayed flat over
4125 frames.

### Plate accuracy

OCR on the [OpenALPR end-to-end EU benchmark](https://github.com/openalpr/benchmarks), 108 labelled
photos, CPU (`python scripts/fetch_assets.py ocr-benchmark`, then `python scripts/eval_ocr.py`):

| Images | Plate read | Exact match | Exact, O/0 and I/1 folded | Char. error rate | Folded CER |
|---|---|---|---|---|---|
| 108 | 97.2% | 90.7% | 97.2% | 0.037 | 0.027 |

Almost every strict error is `0` read as `O`, and the labels themselves are inconsistent about
that pair, so the folded score is shown too. What remains after folding is exactly three vanity
plates without digits, which the letters-and-digits rule rejects on purpose. The benchmark has no
Ukrainian plates.

On the two sample clips the program reports three plates (CF5775, NU04703, KE15NCC), all correct,
and no false ones. The other plates in the dense clip are not readable by eye either.

### Colour accuracy

30 vehicles from both clips, up to six crops each across the life of the track (145 crops),
labelled by hand in `labels/vehicle_colors.csv`. A silver car counts as grey; a colour between
two names, such as dark grey or navy, accepts both. Run with `python scripts/eval_color.py`.

| Per vehicle (majority vote) | Per crop |
|---|---|
| 21 / 30 (70%) | 88 / 145 (61%) |

Wrong: four silver cars read as white, two navy cars and one black car read as grey, a dark red
car read as grey, and a yellow loader on a trailer read as blue.

## Optimisation experiments that were not adopted

Each was measured and dropped because the gain did not justify the cost or the loss of quality.

**ONNX export of the detector.** 18.3 ms against 18.7 ms for PyTorch on the same CPU, boxes
matching at a mean IoU of 0.997. No real gain for an extra build step.

**OpenVINO, FP32 and INT8** (NNCF post-training quantisation calibrated on 300 frames of the
sample clips), full clips on the four-core setup:

| Detector | Dashcam FPS | Dense FPS | Tracks, dense | Plates |
|---|---|---|---|---|
| PyTorch | 14.7 | 11.6 | 49 | all three |
| OpenVINO FP32 | 16.9 (+15%) | 12.9 (+11%) | 49 | all three |
| OpenVINO INT8 | 22.9 (+56%) | 15.8 (+36%) | 55 | KE15NCC lost, false CF5773 |

FP32 gives nothing on the desktop CPU and +11–15% on four cores for a ~40 MB dependency. INT8 is
clearly faster but breaks exactly what the task asks for: an ID per vehicle and the plate.

**ONNX Runtime static INT8** (QDQ, per-channel weights, calibrated on 200 frames of the target
clips, evaluated on 150 others). Quantising the whole model drops recall to 0%: the detection head
does not survive INT8. With the head kept in float it keeps 94.5% of the FP32 boxes at 96.9%
precision and is 1.4× faster than ONNX FP32 on four cores (+6% on the desktop). Losing 5.5% of
the vehicles was too much without accuracy-aware tuning.

**Smaller detector input.** 416 instead of 640 raises the four-core pipeline from 15 to 23 FPS,
but on dense traffic the IDs fragment (64 tracks instead of 49) and KE15NCC is lost.
`--imgsz` is left as an option.

**TensorRT on Jetson.** Not implemented: no hardware to test on, and an engine file is tied to the
GPU and TensorRT version it was built with, so it cannot be shipped in the repository.

## Running on an edge device

What I would do for a real deployment, in this order:

1. **Measure on the target device** with `scripts/benchmark.py`: the numbers above come from a
   desktop.
2. **Inference runtime for the target hardware.** Jetson: export the detector on the device with
   `yolo export model=weights/yolo11n.pt format=engine half=True` and run the plate models through
   `onnxruntime-gpu` (TensorRT or CUDA execution provider). Intel CPU/iGPU: OpenVINO. ARM CPU
   without an accelerator: NCNN or TFLite.
3. **INT8 only with calibration on the target camera**: 500–1000 frames from that camera, a
   labelled validation set checked before and after, sensitive layers such as the detection head
   left in float. If post-training quantisation does not stay within tolerance, quantisation-aware
   training.
4. **Hardware video decoding** (NVDEC or GStreamer on Jetson) for 1080p streams.
5. **Tune `--imgsz` and the sampling intervals** for the camera position and the required frame
   rate, checking tracks and plates on recorded footage from that camera.

## Limitations

- **Colour** is a brightness and hue rule, and brightness depends on lighting: a silver car in sun
  is brighter than a white bus under clouds, so no single threshold separates silver from white
  on both clips (the best one gave 22/30 instead of 21/30, only moving the error from one clip to
  the other). The next step would be a small classifier (for example MobileNetV3) fine-tuned on a
  public vehicle colour dataset on the vehicle crops, with the 30 labelled vehicles as the test
  set. It runs a few times per vehicle, so it would barely affect speed.
- **Detection** uses COCO weights. A dark tow van in the dense clip is missed for a while: it is
  seen as a truck at 0.49 confidence, just under the 0.5 threshold. A roadside sign is sometimes
  boxed as a vehicle; it never gets a plate, because it does not move. Fine-tuning on
  traffic-camera data or a larger model would help.
- **IDs** come from ByteTrack without re-identification: a vehicle hidden for more than about a
  second, or leaving and coming back, gets a new ID.
- **Plates** are read only when they are at least 40 px wide. OCR was benchmarked on European
  plates; the Ukrainian format was checked only on the sample footage.
- **Plate models run on the CPU** with the default `onnxruntime` package, even when a GPU is used
  for detection.
- Tested on Windows 11. The Linux Docker image is built and started in CI, but not benchmarked.

## Development

```bash
pip install -r requirements-dev.txt
pytest              # 195 tests; integration tests need the downloaded models and videos
ruff check . && ruff format --check .
```

Unit tests cover the pure logic (voting, scheduling, colour rules, metrics, config, video source
and writer, evaluation) without the ML stack, so CI runs them on a clean runner. CI also builds the
Docker image and starts it.

```text
vehicle_tracker/
    __main__.py        entry point
    config.py          command line options
    video_source.py    reader thread, stream reconnects
    detector.py        YOLO11n + ByteTrack
    detection.py       detection record, vehicle crop
    tracks.py          per-vehicle state: colour and plate votes, reading schedule
    plate_reader.py    plate detector and OCR
    plate.py           plate text normalisation and vote weights
    color.py           colour rules
    pipeline.py        frame loop, reporting
    overlay.py         boxes and labels
    metrics.py         per-stage timings
    runtime.py         ONNX thread and provider settings, OpenCV GUI check
    video_writer.py    --save-video
    evaluation.py      scoring for the OCR and colour evaluations
scripts/
    fetch_assets.py    models, sample videos, OCR benchmark
    benchmark.py       speed with and without the per-vehicle cache
    eval_ocr.py        plate accuracy on OpenALPR EU
    eval_color.py      colour accuracy on the labelled crops
labels/vehicle_colors.csv
```

## Licence

AGPL-3.0, because the detector is [Ultralytics](https://github.com/ultralytics/ultralytics)
(AGPL-3.0, including the YOLO11 weights). For a closed-source product the detector would have to
be replaced with a permissively licensed one, for example YOLOX (Apache-2.0). The other
dependencies are MIT, BSD or Apache-2.0.

Sample videos are from [Pexels](https://www.pexels.com/license/) and are downloaded by
`fetch_assets.py`, not stored in the repository; so is the OpenALPR benchmark.
