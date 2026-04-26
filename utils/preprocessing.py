from PIL import Image
import torchvision.transforms as transforms
import io


# define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to grayscale
    transforms.Resize((28, 28)),                  # resize to 28x28
    transforms.ToTensor(),                        # convert to tensor
])


def preprocess_image(image_bytes):
    """
    Takes image bytes and returns processed tensor
    """

    # convert bytes → PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # apply transformations
    image = transform(image)

    # add batch dimension (important for model)
    image = image.unsqueeze(0)

    return image