import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# 1. Paths
# ==========================================================

MODEL_PATH = "models/hand_landmarker.task"

IMAGE_PATH = "dataset/ASL_Dataset/dataset/A-samples/0.jpg"


# ==========================================================
# 2. Create MediaPipe Hand Landmarker
# ==========================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================================
# 3. Read Image
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Image could not be loaded.")
    print("Check the image path:")
    print(IMAGE_PATH)

    landmarker.close()
    exit()

print("✅ Image loaded successfully.")


# ==========================================================
# 4. Convert BGR → RGB
# ==========================================================

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================================
# 5. Convert OpenCV image to MediaPipe Image
# ==========================================================

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# ==========================================================
# 6. Detect Hand
# ==========================================================

result = landmarker.detect(mp_image)


# ==========================================================
# 7. Check Hand Detection
# ==========================================================

if len(result.hand_landmarks) == 0:

    print("❌ No hand detected.")

else:

    print("✅ Hand detected!")

    # Get the first detected hand
    landmarks = result.hand_landmarks[0]

    print("Number of landmarks:", len(landmarks))


    # ======================================================
    # 8. Extract x, y, z coordinates
    # ======================================================

    features = []

    for landmark in landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])


    # Convert list to NumPy array
    features = np.array(
        features,
        dtype=np.float32
    )


    # ======================================================
    # 9. Display Feature Information
    # ======================================================

    print("Number of features:", len(features))

    print("\nFirst 10 features:")
    print(features[:10])


# ==========================================================
# 10. Close MediaPipe
# ==========================================================

landmarker.close()

print("\nTest completed.")