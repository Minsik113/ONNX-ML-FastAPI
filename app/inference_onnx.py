# from modeler

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

output_names = ['predictions']
model_path = './resnet50.onnx'
img_path = './etc/bb.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)


# for service engineer

import onnxruntime as rt
providers = ['CUDAExecutionProvider','CPUExecutionProvider'] # GPU 설정시 'CUDAExecutionProvider'
m = rt.InferenceSession(model_path, providers=providers)
onnx_pred = m.run(output_names, {"input": x})
print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])