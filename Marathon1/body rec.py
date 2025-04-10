import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import sys
import tensorflow as tf
from tflite_model_maker import model_spec
# Initialize Mediapipe Holistic and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# Load the image
image_path = "/Users/c27ds/Downloads/lobby8_0101.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load image.")
    sys.exit(1)

# Image dimensions
height, width, _ = image.shape

# Initialize the Object Detector with a person detection model
model_path = 'efficientdet_lite0.tflite'  # Ensure this model file is available
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Convert image to RGB (MediaPipe format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = vision.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Detect people in the image
detection_result = detector.detect(mp_image)

# Process each detected person
for detection in detection_result.detections:
    category = detection.categories[0]
    if category.category_name == 'person' and category.score > 0.5:
        bbox = detection.bounding_box
        xmin, ymin = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height

        # Expand the bounding box (20% padding)
        expand_x = int(w * 0.2)
        expand_y = int(h * 0.2)
        xmin = max(0, xmin - expand_x)
        ymin = max(0, ymin - expand_y)
        xmax = min(width, xmin + w + 2 * expand_x)
        ymax = min(height, ymin + h + 2 * expand_y)
        new_w, new_h = xmax - xmin, ymax - ymin

        # Crop and process with Holistic
        cropped_rgb = image_rgb[ymin:ymax, xmin:xmax]
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, 
                                 min_tracking_confidence=0.5) as holistic:
            results = holistic.process(cropped_rgb)

            # Adjust landmarks to original image coordinates
            def adjust_landmarks(landmarks):
                if not landmarks:
                    return None
                adjusted = []
                for landmark in landmarks.landmark:
                    l = mp.framework.formats.landmark_pb2.NormalizedLandmark()
                    l.x = (landmark.x * new_w + xmin) / width
                    l.y = (landmark.y * new_h + ymin) / height
                    l.z = landmark.z
                    adjusted.append(l)
                adjusted_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                adjusted_landmarks.landmark.extend(adjusted)
                return adjusted_landmarks

            # Get adjusted landmarks
            face = adjust_landmarks(results.face_landmarks)
            left_hand = adjust_landmarks(results.left_hand_landmarks)
            right_hand = adjust_landmarks(results.right_hand_landmarks)
            pose = adjust_landmarks(results.pose_landmarks)

            # Draw adjusted landmarks on the original image
            if face:
                mp_drawing.draw_landmarks(
                    image, face, mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )
            if right_hand:
                mp_drawing.draw_landmarks(
                    image, right_hand, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )
            if left_hand:
                mp_drawing.draw_landmarks(
                    image, left_hand, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )
            if pose:
                mp_drawing.draw_landmarks(
                    image, pose, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

# Display the result
cv2.imshow('Multi-Person Holistic Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()