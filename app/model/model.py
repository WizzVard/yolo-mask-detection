import ultralytics
import json
from typing import List, Dict, Any
from PIL import Image
import cv2
import numpy as np


class YoloInference:
    """
    This class is designed to perform object detection using the YOLOv8 model and then visualize the detection results
    by drawing bounding boxes on the image along with a legend that indicates the classes detected.
    """
    def __init__(self):
        # Load pre-trained weights on the YOLOv8 model
        self.model = ultralytics.YOLO('MaskModel.pt')
        # Define the colors for each class
        self.colors = {
            'without_mask': (0, 0, 255),  # Red color for 'without_mask' (BGR format)
            'with_mask': (0, 255, 0),     # Green color for 'with_mask' (BGR format)
            'mask_weared_incorrect': (255, 0, 0)  # Blue color for 'mask_weared_incorrect' (BGR format)
        }

    def inference_on_img(self, img: Image) -> List[Dict[str, Any]]:
        """
        Runs inference on the YOLOv8 architecture for the given image
        """
        results: ultralytics.engine.reslts.Results = self.model(source=img, show=False, conf=0.45)
        result_data = json.loads(results[0].tojson())

        return result_data

    def draw_bounding_boxes(self, image_path: str, detections: List[Dict[str, Any]]) -> None:
        """Draws bounding boxes on the image and saves it in the same path."""
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            color = self.colors.get(detection['name'], (0, 255, 0))  # Default to green if class not found
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Create a blank image to hold the legend
        legend_height = 100
        legend_image = np.zeros((legend_height, width, 3), dtype=np.uint8)

        # Draw the legend on the blank image
        for i, (label, color) in enumerate(self.colors.items()):
            # Rectangle and text are adjusted for row-based layout
            cv2.rectangle(legend_image, (10, 10 + i * 30), (30, 30 + i * 30), color, -1)
            cv2.putText(legend_image, label, (40, 25 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Concatenate the original image and the legend image vertically
        combined_image = np.vstack((image, legend_image))

        # Save the image with bounding boxes and legend in the same path
        cv2.imwrite(image_path, combined_image)
