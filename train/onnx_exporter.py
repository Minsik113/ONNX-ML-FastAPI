import tf2onnx
import onnxruntime as rt

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)