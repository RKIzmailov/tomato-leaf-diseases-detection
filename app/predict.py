from ultralytics import YOLO

from flask import Flask
from flask import request
from flask import jsonify

import numpy as np
import os
import cv2
from pathlib import Path
import datetime
import yaml

data_path = os.path.join(Path(os.getcwd()),  'data')
data_yaml_path = os.path.join(data_path, 'data.yaml')
with open(data_yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)
class_names = data_yaml['names']

model_path = os.path.join(Path(os.getcwd()),  'model')
output_dir = 'predicted_images'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

model = YOLO(os.path.join(model_path, "best.pt"))

app = Flask('tomato-leaves-disease-detector')

# @app.route('/predict', methods=['POST'])
# def predict(img, model = model):
#     result = model.predict(source=img, save=False)
    
#     #plot the bounding boxes
#     boxes = result.boxes.xyxy
#     classes = result.boxes.cls
#     for box, cls in zip(boxes, classes):
#         x1, y1, x2, y2 = map(int, box)
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         img = cv2.putText(img, class_names[int(cls)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
#     # save the imgage
#     output_path = os.path.join(output_dir, str(today)+".jpg")
#     cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
#     #display the image
#     cv2.imshow("Predicted Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()    
    
#     return jsonify(result)



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Check if an image is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Read image file into memory
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform prediction
    result = model.predict(source=img, save=False)

    # Plot bounding boxes
    boxes = result[0].boxes.xyxy
    classes = result[0].boxes.cls
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        img = cv2.putText(
            img,
            class_names[int(cls)],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # Save the image
    output_path = os.path.join(output_dir, f"{today}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Optionally display the image (comment out in production)
    cv2.imshow("Predicted Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return jsonify({'message': 'Prediction complete', 'output_path': output_path})







if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)