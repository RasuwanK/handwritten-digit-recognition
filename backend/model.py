import torch

model = torch.load("model.pth", map_location=torch.device('cpu'))
model.eval()

def predict_digit(image_array):
    with torch.no_grad():
        tensor = torch.tensor(image_array).float()
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()