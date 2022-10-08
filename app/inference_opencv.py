# from modeler

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

model_path = './resnet50.onnx'
output_names = ['predictions']
img_path = './etc/bb.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)


# for service engineer  opencv 버전 같은 결과물 출력하면 됨

import cv2
opencv_net = cv2.dnn.readNetFromONNX(model_path)
opencv_net.setInput(x)
outputs = opencv_net.forward()
print('ONNX Predicted:', decode_predictions(outputs, top=3)[0])