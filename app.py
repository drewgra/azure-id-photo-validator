import os
import tempfile
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from face_analyser import FaceAnalyzer 
from azure_faces import FacesClient
from azure_content_safety import ContentSafetyClient
from azure_vision import VisionClient

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from dotenv import load_dotenv
load_dotenv()

face_client = FacesClient(os.environ.get("AZURE_FACES_API_ENDPOINT"), os.environ.get("AZURE_FACES_API_KEY"))
content_safety_client = ContentSafetyClient(os.environ.get("AZURE_MODERATION_API_ENDPOINT"), os.environ.get("AZURE_MODERATION_API_KEY"))
vision_client = VisionClient(endpoint=os.environ.get("AZURE_VISION_API_ENDPOINT"), api_key=os.environ.get("AZURE_VISION_API_KEY"))

@app.route('/analyse', methods=['POST'])
def analyse():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        analyser = FaceAnalyzer(temp_path, face_client, content_safety_client, vision_client)
        results = analyser.run_all_tests()
        tags = analyser.get_tags()

        tag_list = [f"Name: {tag.name}, Confidence: {tag.confidence}" for tag in tags.tags.list]

        json = jsonify({"results": results, "tags": tag_list})

        os.remove(temp_path)

        return json

if __name__ == '__main__':
    app.run(debug=True, port=5050)