from azure.core.credentials import AzureKeyCredential

from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceAttributeTypeDetection01, 
    FaceAttributeTypeDetection03, 
    FaceAttributeTypeRecognition04,
    FaceDetectionModel, 
    FaceRecognitionModel
)

class FacesClient:
    def __init__(self, endpoint, api_key):
        self.client = FaceClient(endpoint, AzureKeyCredential(api_key))

    def detect_faces(self, image_stream):
        return self.client.detect(
                image_content=image_stream,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,
                return_face_attributes=[
                    FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION,
                    FaceAttributeTypeDetection01.EXPOSURE,
                    FaceAttributeTypeDetection03.HEAD_POSE,
                    FaceAttributeTypeDetection01.OCCLUSION,
                    FaceAttributeTypeDetection01.BLUR,
                    FaceAttributeTypeDetection03.MASK
                ]
            )