import cv2
import numpy as np
from PIL import Image, ImageStat
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import ImageCategory
from azure.ai.vision.face.models import (
    QualityForRecognition,
    ExposureLevel,
    BlurLevel,
    MaskType
)


# Configuration Constants
MIN_FACE_AREA_RATIO = 0.10
POSITION_TOLERANCE_RATIO = 0.10
MAX_BRIGHTNESS = 200
MIN_BRIGHTNESS = 100
MAX_STDDEV = 60
MAX_YAW = 15
MAX_ROLL = 15
CONFIDENCE_THRESHOLD = 0.5

# Main Face Analyzer Class
class FaceAnalyzer:
    def __init__(self, image_path, face_client, content_safety_client, vision_client):
        self.image_path = image_path
        self.image_cv2, self.image_pil = self.load_image()
        self.image_width, self.image_height = self.image_pil.size
        self.face = None
        self.moderation_result = None
        self.face_client = face_client
        self.content_safety_client = content_safety_client
        self.vision_client = vision_client

    def load_image(self):
        return cv2.imread(self.image_path), Image.open(self.image_path)
    
    def detect_face(self):
        """Detect faces and set the main face attribute for further tests."""
        with open(self.image_path, 'rb') as image_stream:
            if faces := self.face_client.detect_faces(image_stream):
                self.face = faces[0]
                with open(self.image_path, 'rb') as image_stream:
                    try:
                        self.moderation_result = self.content_safety_client.analyze_image(image_stream)
                        self.tags = self.vision_client.get_tags(image_stream)
                    except HttpResponseError as e:
                        print("Analyze image failed.")
                        if e.error:
                            print(f"Error code: {e.error.code}")
                            print(f"Error message: {e.error.message}")
                            raise
                        print(e)
                        raise
                return True
        
        return False
    

    def recognition_quality_test(self):
        quality = self.face.face_attributes.quality_for_recognition
        if quality == QualityForRecognition.HIGH:
            return True, "Recognition quality is high."
        elif quality == QualityForRecognition.MEDIUM:
            return False, "Recognition quality is medium; a higher quality image is recommended."
        return False, "Recognition quality is low; the image is not suitable for identity."

    def face_size_test(self):
        face_width = self.face.face_rectangle.width
        face_height = self.face.face_rectangle.height
        face_area_ratio = (face_width * face_height) / (self.image_width * self.image_height)
        if face_area_ratio >= MIN_FACE_AREA_RATIO:
            return True, f"Face occupies {face_area_ratio:.2%} of the image area."
        return False, f"Face area ratio is too small: {face_area_ratio:.2%}."

    def face_positioning_test(self):
        image_center_x, image_center_y = self.image_width / 2, self.image_height / 2
        face_center_x = self.face.face_rectangle.left + (self.face.face_rectangle.width / 2)
        face_center_y = self.face.face_rectangle.top + (self.face.face_rectangle.height / 2)
        x_offset = abs(image_center_x - face_center_x)
        y_offset = abs(image_center_y - face_center_y)
        if x_offset <= self.image_width * POSITION_TOLERANCE_RATIO and y_offset <= self.image_height * POSITION_TOLERANCE_RATIO:
            return True, "Face is well-centered."
        return False, f"Face position deviates by {x_offset:.1f}px horizontally and {y_offset:.1f}px vertically."

    def lighting_exposure_test(self):
        grayscale_img = self.image_pil.convert("L")
        stat = ImageStat.Stat(grayscale_img)
        brightness = stat.mean[0]

        if brightness < MIN_BRIGHTNESS:
            return False, f"Image is too dark (brightness: {brightness:.1f})."
        elif brightness > MAX_BRIGHTNESS:
            return False, f"Image is too bright (brightness: {brightness:.1f})."
        return True, f"Brightness level is within range: {brightness:.1f}."

    def exposure_test(self):
        exposure = self.face.face_attributes.exposure.exposure_level
        if exposure == ExposureLevel.GOOD_EXPOSURE:
            return True, "Good exposure level detected."
        return False, f"Exposure level issue detected: {exposure}."

    def head_orientation_test(self):
        yaw = self.face.face_attributes.head_pose.yaw
        roll = self.face.face_attributes.head_pose.roll
        if abs(yaw) <= MAX_YAW and abs(roll) <= MAX_ROLL:
            return True, f"Head orientation is within limits (Yaw: {yaw:.1f}, Roll: {roll:.1f})."
        return False, f"Head orientation outside acceptable range (Yaw: {yaw:.1f}, Roll: {roll:.1f})."

    def neutral_background_test(self):
        face_rectangle = self.face.face_rectangle

        """Mask the face area in an image to analyse only the background."""
        height, width, _ = self.image_cv2.shape
        mask = np.ones((height, width), dtype=np.uint8) * 255
        left, top, right, bottom = (
            face_rectangle.left, 
            face_rectangle.top, 
            face_rectangle.left + face_rectangle.width, 
            face_rectangle.top + face_rectangle.height
        )
        cv2.rectangle(mask, (left, top), (right, bottom), 0, -1)

        background = cv2.bitwise_and(self.image_cv2, self.image_cv2, mask=mask)
        gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        _, stddev = cv2.meanStdDev(gray_background, mask=mask)
        
        if stddev[0][0] <= MAX_STDDEV:
            return True, "Background is uniform and neutral."
        return False, f"Background variation too high (stddev = {stddev[0][0]:.2f})."

    def blur_test(self):
        """Check the blur level of the face."""
        blur = self.face.face_attributes.blur
        if blur.blur_level == BlurLevel.LOW:
            return True, "Blur test passed. Blur level is low."
        elif blur.blur_level == BlurLevel.MEDIUM:
            return True, "Blur test passed. Blur level is medium."
        else:  # BlurLevel.HIGH
            return True, "Blur test ignored. Blur level is high; the image is too blurry."

    def mask_test(self):
        """Check if the face has a mask that could obscure identity features."""
        mask = self.face.face_attributes.mask
        if mask.type == MaskType.NO_MASK:
            return True, "Mask test passed. No mask detected."
        else:
            return False, f"Mask test failed. Detected mask type: {mask.type}."

    def occlusion_test(self):
        """Check if the face has a mask that could obscure identity features."""
        occlusion = self.face.face_attributes.occlusion
        occlusions = []
        if occlusion.eye_occluded:
            occlusions.append("eye")

        if occlusion.forehead_occluded:
            occlusions.append("forehead")

        if occlusion.mouth_occluded:
            occlusions.append("mouth")

        if len(occlusions) > 0:
            return False, f"Occlusion test failed. Occlusions: {','.join(occlusions)}."

        return True, "Occlusion test passed. No occlusions detected."

    def moderate_image(self):
        hate_result = next(item for item in self.moderation_result.categories_analysis if item.category == ImageCategory.HATE)
        self_harm_result = next(item for item in self.moderation_result.categories_analysis if item.category == ImageCategory.SELF_HARM)
        sexual_result = next(item for item in self.moderation_result.categories_analysis if item.category == ImageCategory.SEXUAL)
        violence_result = next(item for item in self.moderation_result.categories_analysis if item.category == ImageCategory.VIOLENCE)

        moderation_results = []

        if hate_result.severity > 0:
            moderation_results.append(f"hate: {hate_result.severity}")
        if self_harm_result.severity > 0:
            moderation_results.append(f"self harm: {self_harm_result.severity}")
        if sexual_result.severity > 0:
            moderation_results.append(f"sexual: {sexual_result.severity}")
        if violence_result.severity > 0:
            moderation_results.append(f"violence: {violence_result.severity}")

        if len(moderation_results)>0:
            return False, f"Moderation test failed. Results: {", ".join(moderation_results)}"
        
        return True, "Moderation test passed"

    # Aggregator Method for Running All Tests
    def run_all_tests(self):
        
        if not self.detect_face():
            return False, "No face detected. Ensure a clear, single face in the image."
        tests = [
            self.recognition_quality_test(),
            self.face_size_test(),
            self.face_positioning_test(),
            self.lighting_exposure_test(),
            self.exposure_test(),
            self.head_orientation_test(),
            self.neutral_background_test(),
            self.blur_test(),
            self.mask_test(),
            self.occlusion_test(),
            self.moderate_image()
        ]
        results = {"status": True, "messages": []}
        for passed, message in tests:
            results["messages"].append(message)
            if not passed:
                results["status"] = False
        return results
    
    def get_tags(self):
        return self.tags