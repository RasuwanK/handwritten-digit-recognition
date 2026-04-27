import sys
import os
import asyncio

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app, startup_event

client = TestClient(app)

async def run_tests():
    print("Running startup event to load models...")
    await startup_event()
    
    print("Testing /generate endpoint...")
    response = client.get("/generate")
    assert response.status_code == 200
    data = response.json()
    assert "image" in data
    assert data["image"].startswith("data:image/png;base64,")
    print("/generate passed.")
    
    print("Testing /detect endpoint with a generated image...")
    # Send the generated image to the detect endpoint
    detect_payload = {
        "image": data["image"]
    }
    response_detect = client.post("/detect", json=detect_payload)
    assert response_detect.status_code == 200
    detect_data = response_detect.json()
    assert "prediction" in detect_data
    assert "confidence" in detect_data
    print(f"/detect passed. Prediction: {detect_data['prediction']}, Confidence: {detect_data['confidence']:.4f}")
    
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(run_tests())
