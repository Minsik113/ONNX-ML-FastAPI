import os
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
import onnxruntime

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = './etc/bb.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

model = ResNet50(weights='imagenet')
preds = model.predict(x)
print('Keras Predicted:', decode_predictions(preds, top=3)[0]) 
model.save(os.path.join("./model", model.name))