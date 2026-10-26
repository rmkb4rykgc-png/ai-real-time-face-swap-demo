import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

WINDOW_NAME = "AI Real-Time Face Swap Demo"
SOURCE_IMAGE_PATH = Path("source.jpg")
SHOW_DEBUG = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def detect_face_points(image_bgr):
    """Return 468 face-landmark points as an array of (x, y) coordinates."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    return points


def build_face_mask(shape, points):
    """Create a convex-hull mask for the face."""
    h, w = shape[:2]
    hull = cv2.convexHull(points.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # Slightly expand the mask to reduce hard-edge seams.
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def warp_source_face(source_bgr, source_points, target_points, target_shape):
    """Warp the source face into the target face position."""
    h, w = target_shape[:2]
    matrix, _ = cv2.estimateAffinePartial2D(source_points, target_points)
    if matrix is None:
        return None, None

    warped_face = cv2.warpAffine(
        source_bgr,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    source_mask = build_face_mask(source_bgr.shape[:2], source_points)
    warped_mask = cv2.warpAffine(
        source_mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped_face, warped_mask


def main():
    if not SOURCE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Missing source image: {SOURCE_IMAGE_PATH}. "
            "Please add a clear front-facing portrait as source.jpg in the project folder."
        )

    source_bgr = cv2.imread(str(SOURCE_IMAGE_PATH))
    if source_bgr is None:
        raise ValueError(f"Could not read the source image: {SOURCE_IMAGE_PATH}")

    source_points = detect_face_points(source_bgr)
    if source_points is None:
        raise RuntimeError(
            "The source image does not contain a detectable face. "
            "Please use a clear front-facing portrait."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Please make sure your camera is connected.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read webcam frame.")
            break

        target_points = detect_face_points(frame)
        if target_points is not None:
            warped_face, warped_mask = warp_source_face(
                source_bgr,
                source_points,
                target_points,
                frame.shape,
            )

            if warped_face is not None and warped_mask is not None:
                target_face_mask = build_face_mask(frame.shape, target_points)
                final_mask = cv2.bitwise_and(target_face_mask, warped_mask)

                if cv2.countNonZero(final_mask) > 0:
                    result = frame.copy()
                    result[final_mask > 0] = warped_face[final_mask > 0]

                    # Soft-edge blending to avoid harsh seams.
                    blur = cv2.GaussianBlur(final_mask.astype(np.float32), (31, 31), 0)
                    alpha = np.clip(blur / 255.0, 0, 1)
                    alpha = cv2.cvtColor(alpha.astype(np.float32), cv2.COLOR_GRAY2BGR)

                    frame = cv2.convertScaleAbs(
                        (frame.astype(np.float32) * (1.0 - alpha))
                        + (result.astype(np.float32) * alpha)
                    )

        if SHOW_DEBUG and target_points is not None:
            for point in target_points:
                x, y = map(int, point)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

