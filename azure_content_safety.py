from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData

class ContentSafetyClient:
    def __init__(self, endpoint, api_key):
        self.client = ContentSafetyClient(endpoint, AzureKeyCredential(api_key))

    def analyse_content(self, image_stream):
        request = AnalyzeImageOptions(image=ImageData(content=image_stream.read()))
        return self.client.analyze_image(request)