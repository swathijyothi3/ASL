import cv2

from utils.predictor import ASLPredictor


# Create predictor
predictor = ASLPredictor()

print("✅ Predictor loaded successfully!")


# Test image
IMAGE_PATH = "dataset/ASL_Dataset/dataset/A-samples/0.jpg"

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("❌ Image could not be loaded.")
    predictor.close()
    exit()


# Predict
letter, confidence = predictor.predict(frame)


# Display result
if letter is None:

    print("❌ No hand detected.")

else:

    print("================================")
    print("Predicted letter:", letter)
    print(f"Confidence: {confidence:.2f}%")
    print("================================")


# Close MediaPipe
predictor.close()

print("✅ Test completed!")