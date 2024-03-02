from flask import Flask, render_template, request
from PIL import Image
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus as CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models.resnet import resnet18 as ResNetModel
from torchvision.models.resnet import BasicBlock
from torchvision.models.mobilenetv3 import mobilenet_v3_large as MobileNetV3Model
from werkzeug.utils import secure_filename
import imghdr

app = Flask(__name__)
plt.switch_backend('Agg')

IMAGE_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_and_save_heatmap(image_path, model, combined_figure_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)

    # Move the model and input image to the correct device
    model.to(DEVICE)
    input_image = input_image.to(DEVICE)

    # Predict the class
    output = model(input_image)
    predicted_class = output.argmax(dim=1).item()

    # Generate GradCAMPlusPlus heatmap
    cam = CAM(model=model, target_layers=[model.layer3[-1]])
    target = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_image, targets=target)

    # Convert to RGB and overlay on the original image
    visualization = show_cam_on_image(
        input_image.cpu().numpy().squeeze().transpose(1, 2, 0),
        grayscale_cam[0, :],
        use_rgb=True
    )

    # Display the original image, prediction, and segmented image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('Segmentation Heatmap')


    # Remove x and y axis labels
    plt.subplot(1, 2, 1).set_xticks([])
    plt.subplot(1, 2, 1).set_yticks([])
    plt.subplot(1, 2, 2).set_xticks([])
    plt.subplot(1, 2, 2).set_yticks([])

    # Save the combined figure
    plt.savefig(combined_figure_path, bbox_inches='tight')

# Usage example for segmentation
model_seg = ResNetModel()
model_seg.fc = torch.nn.Linear(in_features=model_seg.fc.in_features, out_features=4)
model_seg.load_state_dict(torch.load('best_model-2.pth', map_location=DEVICE))
model_seg.eval()

def predict_tumor_class(image_path):
    # Define the classes list
    classes = ['Glioma Tumor', 'Meningioma Tumor', 'Normal', 'Pituitary Tumor']

    # Specify the path to the saved model
    model_path = "best_model.pth"  # Assuming the PyTorch model is saved as .pth

    # Load the trained model
    model_det = MobileNetV3Model()
    model_det.classifier[3] = torch.nn.Linear(in_features=model_det.classifier[3].in_features, out_features=len(classes))

    # Use GPU if available
    device_det = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_det.load_state_dict(torch.load(model_path, map_location=device_det))
    model_det.to(device_det)
    model_det.eval()

    # Load and preprocess the image
    transform_val = transforms.Compose([transforms.ToTensor()])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return "Error processing the image. Please make sure it's a valid image file."

    try:
        img = transform_val(img).unsqueeze(0).to(device_det)
    except Exception as e:
        return "Error processing the image. Please try again."

    # Make predictions
    with torch.no_grad():
        output_det = model_det(img)
        probabilities = torch.nn.functional.softmax(output_det, dim=-1)
        predicted_class_index = torch.argmax(probabilities, dim=-1).item()

    # Check if the prediction is empty
    if not classes:
        return "Error: Classes list is empty."

    # Decode predictions
    class_label = classes[predicted_class_index]
    return class_label


def convert_file_path(original_path):
    # Replace backslashes with forward slashes
    converted_path = original_path.replace("\\", "/")
    
    # Remove the 'static/' prefix if present
    if converted_path.startswith("static/"):
        converted_path = converted_path[len("static/"):]

    return converted_path

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        imagefile = request.files['imagefile']
        image_path = os.path.join("static","upload",secure_filename(imagefile.filename))

        try:
            imagefile.save(image_path)
            if imghdr.what(image_path) not in ['jpeg', 'png', 'gif', 'bmp']:
                raise ValueError("Invalid image format")
        except Exception as e:
            print("Error saving or validating the image:", e)
            return render_template('index.html', error="Error saving or validating the image. Please try again."), 400

        try:
            predicted_class = predict_tumor_class(image_path)
        except Exception as e:
            print("Error predicting tumor class:", e)
            return render_template('index.html', error="Error predicting tumor class. Please try again."), 400

        # Generate heatmap
        generate_and_save_heatmap(image_path, model_seg, image_path)
        converted_path = convert_file_path(image_path)
        return render_template('result.html', prediction=predicted_class, image_path=converted_path)


    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)

