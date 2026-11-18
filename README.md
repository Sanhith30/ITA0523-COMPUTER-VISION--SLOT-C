# ITA0523 Computer Vision — Slot C

This repository contains the Computer Vision coursework for **ITA0523**, including 80 lab experiments, five assessment submissions, and a gesture-controlled game Capstone project.

## Repository structure

```text
.
├── Assesments/
│   ├── CO1/              Assignment 1 PDF
│   ├── CO2/              Assignment 2 PDF
│   ├── CO3/              Assignment 3 PDF
│   ├── CO4/              Assignment 4 PDF
│   └── CO5/              Assignment 5 PDF and README
├── Capstone/
│   ├── main.py           Gesture-controlled game application
│   ├── capture.py        Camera/capture utility
│   ├── test_camera.py    Camera test
│   ├── requirements.txt  Python dependencies
│   ├── config/           Configuration and key mappings
│   └── src/              Hand tracking, gesture logic, and game control
└── Lab Experiments/
    ├── Exp01/ ... Exp80/ Individual experiment folders
    ├── Readme.md         Complete experiment manual
    └── *.pdf             Compiled experiment record
```

> The existing directory is named `Assesments` in the repository and is kept unchanged for compatibility.

## Lab Experiments

`Lab Experiments/Exp01` through `Lab Experiments/Exp80` contain individual OpenCV and Python exercises. Most experiment folders include:

- `main.py` — experiment implementation
- `input.*` — source image or video, when required
- `output.*` — generated result or visualization
- Optional templates, extracted text, or processed media

The experiments cover Gaussian smoothing, histogram processing, image rotation and resizing, Sobel and Canny edge detection, thresholding, morphology, affine and perspective transformations, region-of-interest processing, face and feature detection, object recognition, OCR, video processing, and related computer-vision tasks.

See [`Lab Experiments/Readme.md`](Lab%20Experiments/Readme.md) for the detailed aims, procedures, code, inputs, outputs, and results for all 80 experiments.

To run an experiment:

```bash
cd "Lab Experiments/Exp01"
python main.py
```

Replace `Exp01` with the experiment folder you want to run. Some experiments require a local input image, video, camera, or additional system packages.

## Assessments

The `Assesments` directory contains the coursework submissions organized by course outcome:

| Folder | Contents |
|---|---|
| `CO1` | Assignment 1 |
| `CO2` | Assignment 2 |
| `CO3` | Assignment 3 |
| `CO4` | Assignment 4 |
| `CO5` | Assignment 5 and supporting README |

## Capstone

The Capstone is a real-time hand-gesture controller. It uses a camera and MediaPipe hand landmarks to detect gestures and map them to keyboard/game actions.

### Setup

```bash
cd Capstone
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The project requires Python, OpenCV, MediaPipe, PyAutoGUI, and NumPy. Camera access and permission for simulated keyboard input may be required.

### Run

```bash
cd Capstone
python main.py
```

Press **Esc** in the OpenCV window to stop the application.

## Notes

- Generated images, videos, PDFs, and experiment outputs are intentionally stored with the coursework.
- Run scripts from their own experiment or project directory so relative input paths resolve correctly.
- Python cache files such as `__pycache__` are implementation artifacts and are not required to understand the source code.
