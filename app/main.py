from tensorflow.keras.applications.resnet50 import decode_predictions

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing import image
import io
from PIL import Image


import onnxruntime as rt
import numpy as np

# providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']

model_path = './resnet50.onnx'
output_names = ['predictions']
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(model_path, providers=providers)


from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()


@app.post("/files")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

@app.post("/predict")
async def predict_img(img: UploadFile, count:int = Form()):
    img_buffer = await img.read()

    # dnn_img = image.load_img(img_buffer, target_size=(224, 224))
    dnn_img = Image.open(io.BytesIO(img_buffer))
    dnn_img = dnn_img.resize((224,224))

    x = image.img_to_array(dnn_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    onnx_pred = m.run(output_names, {"input": x})
    results = decode_predictions(onnx_pred[0], top=count)[0]
    print(results)    
    result_data = [{'name':result[1],'score':result[2].item()} for result in results]
    print(result_data)
    return result_data 