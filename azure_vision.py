from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

class VisionClient:
    def __init__(self, endpoint, api_key):
        self.client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )    
    
    def get_tags(self, image_stream):
        result = self.client.analyze(
            image_data=image_stream.read(),
            visual_features=[VisualFeatures.TAGS, VisualFeatures.PEOPLE, VisualFeatures.CAPTION, VisualFeatures.OBJECTS, VisualFeatures.SMART_CROPS],
            language="en",
        )

        return result