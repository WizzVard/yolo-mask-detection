from typing import Dict, Union, Any
from main import app
from model.model import YoloInference
from fastapi import Request
from PIL import Image
from utils.load_config import LoadConfig
import os

CONFIG = LoadConfig()

yolo_inference = YoloInference()


@app.get('/')
def home() -> Dict[str, Union[int, Dict[str, Any]]]:
    """Ping method for checking API status"""
    return {
        'status_code': 200,
        'data': {'api_health_status': 'OK'}
    }


@app.post('/detect_image')
async def detect_img(request: Request) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Runs YOLO inference for the image received"""

    data = await request.json()

    # Change inference folder
    inference_folder = os.path.join("./app", CONFIG.inference_folder)
    # Get image path with corrected slashes
    temp_image_path = os.path.join(inference_folder, data.get('image_path')).replace("\\", "/")

    if not temp_image_path:
        return {'status_code': 400, 'data': 'Missing temp_image_path'}

    # Run inference on the received image
    try:
        image = Image.open(temp_image_path)
        inference_results_data = yolo_inference.inference_on_img(img=image)

        # Draw bounding boxes and save the image (in the same path or a new one)
        yolo_inference.draw_bounding_boxes(temp_image_path, inference_results_data)

    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {
            'status_code': 500, 'data': {}
        }

    # Extract the filename and image size from the image
    image_name = os.path.basename(temp_image_path)
    image_size = image.size

    return {
        'status_code': 200,
        'data': {'image_name': image_name,
                 'image_size': image_size,
                 'inference_results': inference_results_data}
    }

