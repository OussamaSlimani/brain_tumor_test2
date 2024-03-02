from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus as CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models.resnet import resnet18 as ResNetModel
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.mobilenetv3 import mobilenet_v3_large as MobileNetV3Model

#################### CODE FOR SEGMENTATION #################

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

    plt.show()

    print(f"Combined figure saved at: {combined_figure_path}")


IMAGE_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Usage example for segmentation
model_seg = ResNetModel()
model_seg.fc = torch.nn.Linear(in_features=model_seg.fc.in_features, out_features=4)
model_seg.load_state_dict(torch.load('best_model-2.pth', map_location=DEVICE))
model_seg.eval()

image_path_seg = 'b.jpg'
combined_figure_path_seg = 'images/combined_figure.png'

generate_and_save_heatmap(image_path_seg, model_seg, combined_figure_path_seg)

#################### CODE FOR DETECTION #################

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
        print("Error opening the image:", e)
        return "Error processing the image. Please make sure it's a valid image file."

    try:
        img = transform_val(img).unsqueeze(0).to(device_det)
    except Exception as e:
        print("Error processing the image:", e)
        return "Error processing the image. Please try again."

    # Add debug statements
    print("Image path:", image_path)
    print("Image shape:", img.shape)

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

image_path_det = "a.jpg"

# Call the prediction function
predicted_class_det = predict_tumor_class(image_path_det)
print("Predicted class:", predicted_class_det)
