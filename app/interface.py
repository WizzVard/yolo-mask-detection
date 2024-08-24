import gradio as gr
import requests
import os
from PIL import Image, ImageOps
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

    # Introduce a delay to prevent loading more than one image within 0.5 seconds
    time.sleep(0.5)

    image_path = os.path.join(CONFIG.inference_folder, "temp_image.png")
    image.save(image_path)

    # Send the image path to FastAPI for processing
    response = requests.post(f"{CONFIG.fastapi_endpoint}/detect_image", json={"image_path": "temp_image.png"})

    if response.status_code == 200:
        # Load and return the processed image
        processed_image = Image.open(image_path)

        # Resize the processed image to scale it up (e.g., double the size)
        processed_image = ImageOps.scale(processed_image, 2)

        # Extract class names from the response
        inference_results = response.json().get('data', {}).get('inference_results', [])
        num_classes = len(inference_results)  # Get the number of detected classes

        # Return the processed image and the number of detected classes
        return processed_image, f"Number of detected classes: {num_classes}"
    else:
        return "Error: Failed to process the image."


# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Image(type="pil", label="Output Image"), gr.Textbox(label="Number of Detected Classes")],
    title="Mask Detection With YOLO",
    description="Upload an image to detect a mask."
)

iface.launch()
