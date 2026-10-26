# AI Real-Time Face Swap Demo

This project is a lightweight Python demo that captures your webcam feed and swaps in a source portrait in real time.

Features:
- Webcam input
- Real-time face detection using MediaPipe Face Mesh
- Face alignment and mask-based blending
- Source portrait image driven by `source.jpg`

Important:
- This is a learning/demo project.
- Face replacement quality is limited by the simple affine mask approach.
- Use only for local testing and legal, consent-based scenarios.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place a clear portrait image in the project folder and name it:

```bash
source.jpg
```

Use a front-facing image with good lighting and no heavy occlusion.

3. Run the app:

```bash
python face_swap_demo.py
```

4. Press `q` to quit.

## Notes

- If the source face is not detected, try using a cleaner portrait.
- If the webcam is not recognized, check that your camera is connected and not used by another app.
- This is not a production-grade deepfake model; it is a real-time portrait replacement prototype.

## Recommended source image

The source image should ideally:
- be a single clear face
- have neutral expression
- be well lit
- avoid sunglasses, masks, and heavy shadows

