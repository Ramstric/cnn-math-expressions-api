from flask import Flask, request, jsonify
from flask_cors import CORS

import os

from flask import send_from_directory

import image_segmentation
import cnn_prediction
app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        response = jsonify({'error': 'No selected file'})
        return response

    # Clear the directory
    for f in os.listdir('data/uploads'):
        os.remove(os.path.join('data/uploads', f))

    # Save the uploaded file (for demonstration purposes)
    file_path = 'data/uploads/' + file.filename
    file.save(file_path)

    response = jsonify({'message': 'File uploaded successfully!'})
    response.headers.add('Access-Control-Allow-Origin', '*')

    #segment_image()
    return response

@app.route('/process', methods=['GET'])
def process_image():
    try:
        image_segmentation.segment_upload()
    finally:
        num_segments = len(os.listdir('data/equation_segmented'))
        response = jsonify({'message': 'Image processed successfully!', 'num_segments': num_segments})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/download', methods=['GET'])
def download_image():
    image_option = request.args.get('image')

    if image_option == 'processed':
        return send_from_directory('data/equation_processed', 'processed.png')
    elif image_option == 'contours':
        return send_from_directory('data/equation_processed', 'contours.png')
    elif image_option == 'segmented':
        segment_number = request.args.get('n')
        return send_from_directory('data/equation_segmented', f'segmented_{segment_number}.png')


@app.route('/predict', methods=['GET'])
def predict():
    latex_code, py_function_code = cnn_prediction.predict()

    response = jsonify({'latex': latex_code, 'python': py_function_code})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
