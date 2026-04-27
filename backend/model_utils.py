import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms

def base64_to_image(base64_str: str) -> Image.Image:
    """Converts a base64 string to a PIL Image."""
    # Remove prefix if present (e.g., "data:image/png;base64,")
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def preprocess_for_classification(image: Image.Image) -> torch.Tensor:
    """Preprocesses a PIL Image for DeformableNet classification."""
    # Ensure image is grayscale
    image = image.convert("L")
    
    # We want white digits on a black background.
    # Often HTML canvas gives black drawings on transparent/white background.
    # We may need to invert if the background is predominantly white.
    # A simple trick is to check corner pixel. Let's assume standard white-on-black for now.
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(image).unsqueeze(0) # Add batch dimension -> (1, 1, 28, 28)
    return tensor
