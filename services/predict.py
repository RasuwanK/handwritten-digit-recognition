import torch

#adjust the path as nedded 
model = torch.load("models/digit_model.pth", map_location=torch.device('cpu'))
model.eval()

def predict_digit(processed_image):

    with torch.no_grad():
        output = model(processed_image)
        predicted = output.argmax(dim=1).item()

    return predicted

