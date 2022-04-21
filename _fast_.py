import warnings
warnings.simplefilter(action='ignore', category=Warning)
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse, JSONResponse
import numpy as np
import sys, cv2, uvicorn
sys.path.append(".")
from getway import getway
from fastapi.encoders import jsonable_encoder

model = getway(config_file='config.ini')
model.load_model(weight_path=model.config.get("model","pre_train_model_path"))
app = FastAPI()
@app.get("/")
def root():
    print("API is Running")
    return {"API is Running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image as a stream of bytes
    file_bytes = await file.read()
    file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print("Image type: ", type(image), "Image Shape: ",image.shape)
    img_4d = image.reshape(-1,image.shape[0],image.shape[1],image.shape[2])
    pred_box = model.predict(img_4d)

    print(pred_box)
    json_compatible_item_data = jsonable_encoder({"boxes": pred_box[0].tolist(), "scores": pred_box[1].tolist(), "class": pred_box[2].tolist()})
    return JSONResponse(content=json_compatible_item_data)


@app.post("/uploader/")
async def uploader(file: UploadFile):
    f = await file.read()
    print("Start uploading file ...")

    with open("./model_data/model.h5", "wb") as file:
        file.write(f)
        file.close()
    model.load_model(weight_path="./model_data/model.h5")
    return {"Upload model"}

if __name__ == "__main__":
    uvicorn.run("_fast_:app", host="0.0.0.0", port=2222, reload=True)
