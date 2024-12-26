from ultralytics import YOLO

import cv2
import os
import yaml
from pathlib import Path

path = os.path.join(Path(os.getcwd()),  'data')

data_yaml_path = os.path.join(path, 'data.yaml')
with open(data_yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)

nc = data_yaml['nc']
class_names = data_yaml['names']

train_path = os.path.join(path, 'train', 'images')
val_path = os.path.join(path, 'valid', 'images')
test_path = os.path.join(path, 'test', 'images')

train_labels_path = os.path.join(path, 'train', 'labels')
val_labels_path = os.path.join(path, 'valid', 'labels')
test_labels_path = os.path.join(path, 'test', 'labels')

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def load_labels_with_image_paths(images_path, labels_path):
    images, labels = [], []
    for label_file in Path(labels_path).glob('*.txt'):
        with open(label_file, 'r') as file:
            label_lines = file.readlines()
        if label_lines:
            img_path = images_path +'/'+ (label_file.stem + '.jpg')
            images.append(str(img_path))
            labels.append([list(map(float, line.split())) for line in label_lines])
    return images, labels

def trian():
    train_images, train_labels = load_labels_with_image_paths(train_path, train_labels_path)
    val_images, val_labels = load_labels_with_image_paths(val_path, val_labels_path)
    test_images, test_labels = load_labels_with_image_paths(test_path, test_labels_path)

    model = YOLO("yolo11n.pt")
    print("""
    @software{yolo11_ultralytics,
    author = {Glenn Jocher and Jing Qiu},
    title = {Ultralytics YOLO11},
    version = {11.0.0},
    year = {2024},
    url = {https://github.com/ultralytics/ultralytics},
    orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
        license = {AGPL-3.0}
    }""")


    result = model.train(data=data_yaml_path, epochs=100, batch = 96, imgsz=640, lr0=0.01, name='train', seed = 42, plots = True)

    print('Training complete. Results saved to "/runs/detect/train/weights/best.pt"')
    
if __name__ == '__main__':
    trian()