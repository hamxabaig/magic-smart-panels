import os
import json
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

def download_image(image_url):
    """Download image from a given URL and return it as a NumPy array."""
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise ValueError("Failed to download image")

    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def detect_panels(image, panels_to_detect):
    """Detect rectangular panels in an image and return their coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add a border to the image
    border_size = 50
    gray_padded = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)

    # Apply Gaussian Blur
    gray_blurred = cv2.GaussianBlur(gray_padded, (5, 5), 0)

    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Edge Detection
    edges = cv2.Canny(binary, threshold1=50, threshold2=150)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes and filter by size
    min_area_threshold = 10000
    panels = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]

    # Adjust for border offset
    panels = [(x - border_size, y - border_size, w, h) for x, y, w, h in panels]

    # Sort by area and return only the top 'panels_to_detect'
    panels = sorted(panels, key=lambda x: x[2] * x[3], reverse=True)[:panels_to_detect]

    return panels

@app.route('/detect_panels', methods=['POST'])
def lambda_handler():
    """AWS Lambda function to process image and return panel coordinates."""
    data = request.get_json()
    image_url = data.get("imageUrl")
    panels_to_detect = int(data.get("panelsToDetect", 3))

    if not image_url:
        return jsonify({"error": "Missing 'imageUrl' parameter"}), 400

    try:
        # Download and process image
        image = download_image(image_url)
        panels = detect_panels(image, panels_to_detect)
        return jsonify({"panels": panels})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
