from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from model import predict_digit  

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("L")

        img = img.resize((28, 28))

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 1, 28, 28)  

        digit = predict_digit(img_array)

        return JSONResponse(content={"digit": int(digit)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)