import ultralytics
import json
from typing import List, Dict, Any
from PIL import Image
import cv2


# Load pre-trained weights on the YOLOv8 model
model = ultralytics.YOLO('yolov8m.pt')


def inference_on_img(img: Image) -> List[Dict[str, Any]]:
    """
    Runs inference on the YOLOv8 architecture for the given image
    """
    results: ultralytics.engine.reslts.Results = model(source=img, show=False, conf=0.45)
    result_data = json.loads(results[0].tojson())

    return result_data


def draw_bounding_boxes(image_path: str, detections: List[Dict[str, Any]]) -> None:
    """Draws bounding boxes on the image and saves it in the same path."""
    image = cv2.imread(image_path)
    for detection in detections:
        box = detection['box']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        label = f"{detection['name']} {detection['confidence']:.2f}"
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes in the same path
    cv2.imwrite(image_path, image)
