from pydantic import BaseModel, Field

class DetectRequest(BaseModel):
    # base64 encoded string of the image
    image: str = Field(..., description="Base64 encoded string of the handwritten digit image")

class DetectResponse(BaseModel):
    prediction: int = Field(..., description="The predicted digit (0-9)")
    confidence: float = Field(..., description="The confidence score of the prediction (0-1)")

class GenerateRequest(BaseModel):
    text: str = Field(..., description="A string of digits to generate images for (e.g., '123')")

class GenerateResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded string of the generated digit image")
