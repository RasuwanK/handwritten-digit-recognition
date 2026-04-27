import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add the root directory to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.build_classifier import DeformableNet
from utils.build_generator import Generator
from backend.schemas import DetectRequest, DetectResponse, GenerateResponse
from backend.model_utils import base64_to_image, image_to_base64, preprocess_for_classification

app = FastAPI(title="Handwritten Digit Recognition API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = None
generator = None

@app.on_event("startup")
async def startup_event():
    global classifier, generator
    
    # Paths to the saved models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier_path = os.path.join(base_dir, "models", "classifier", "mnist_deformable_net.pth")
    generator_path = os.path.join(base_dir, "models", "generator", "final_generator.pth")
    
    try:
        # Load Classifier
        classifier = DeformableNet().to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
        classifier.eval()
        print("Classifier loaded successfully.")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        
    try:
        # Load Generator
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
        generator.eval()
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"Error loading generator: {e}")

@app.post("/detect", response_model=DetectResponse)
async def detect_digit(request: DetectRequest):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier model not loaded.")
        
    try:
        image = base64_to_image(request.image)
        tensor = preprocess_for_classification(image).to(device)
        
        with torch.no_grad():
            outputs = classifier(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return DetectResponse(
            prediction=int(predicted.item()),
            confidence=float(confidence.item())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/generate", response_model=GenerateResponse)
async def generate_digit():
    """Generates a random handwritten digit."""
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator model not loaded.")
        
    try:
        # Generator expects 100-dim noise
        noise = torch.randn(1, 100, 1, 1, device=device)
        
        with torch.no_grad():
            fake_image = generator(noise) # Shape: (1, 1, 32, 32)
            
        # The generator has a Tanh output, so images are in range [-1, 1].
        # We need to scale them back to [0, 1] then [0, 255] for PIL.
        # torchvision.utils.make_grid with normalize=True does scaling to [0, 1]
        grid = vutils.make_grid(fake_image, normalize=True)
        # grid shape is (3, 32, 32)
        
        # Convert tensor to numpy and then to PIL Image
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        
        # Convert PIL Image to Base64
        b64_str = image_to_base64(image)
        
        return GenerateResponse(image=b64_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating digit: {str(e)}")
