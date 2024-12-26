import requests

import numpy as np
import os
import cv2
from pathlib import Path


url = 'http://localhost:9696/predict'

data_path = os.path.join(Path(os.getcwd()),  'data')
test_path = os.path.join(data_path, 'test', 'images')
test_labels_path = os.path.join(data_path, 'test', 'labels')
images_list = os.listdir(test_path)

img_idx = np.random.randint(0, len(images_list))
random_img_name = images_list[img_idx]


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


img = load_image(os.path.join(test_path, random_img_name))
# response = requests.post(url, img=img).json()
with open(os.path.join(test_path, random_img_name), 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")