import gradio as gr
import requests
import os
from PIL import Image
from app.utils.load_config import LoadConfig
import time

CONFIG = LoadConfig()


def process_image(image):
    """
    This function processes an input image by saving it locally, sending it to a FastAPI endpoint for
    further processing, and returning the processed image.
    :param image: (PIL.Image): The image object uploaded by the user.
    :return:
    """
    image_path = os.path.join(CONFIG.inference_folder, "temp_image.png")
    time.sleep(0.1)
    image.save(image_path)

    # Send the image path to FastAPI for processing
    response = requests.post(f"{CONFIG.fastapi_endpoint}/detect_image", json={"image_path": "temp_image.png"})

    if response.status_code == 200:
        # Load and return the processed image
        return Image.open(image_path)
    else:
        return "Error: Failed to process the image."


# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil"),
    title="Object Detection With Yolo v8",
    description="Upload an image to detect an object."
)

iface.launch()
