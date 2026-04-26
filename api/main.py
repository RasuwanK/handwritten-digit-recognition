from fastapi import FastAPI, UploadFile, File
from utils.preprocessing import preprocess_image
from services.predict import predict_digit


app = FastAPI()


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # read image
    image_data = await image.read()

    # preprocess image
    processed_image = preprocess_image(image_data)

    # predict digit
    digit = predict_digit(processed_image)

    return {
        "message": "Image processed successfully",
        "shape": str(processed_image.shape),
        "digit": digit
    }