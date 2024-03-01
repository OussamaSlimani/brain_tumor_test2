from flask import Flask, render_template, request
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_large as Model

app = Flask(__name__)

def predict_tumor_class(image_path):
    # Define the classes list
    classes = ['Glioma Tumor', 'Meningioma Tumor', 'Normal', 'Pituitary Tumor']

    # Specify the path to the saved model
    model_path = "best_model.pth"  # Assuming the PyTorch model is saved as .pth

    # Load the trained model
    model = Model(weights=None)  # Initialize MobileNetV3 model
    model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=len(classes))

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess the image
    transform_val = transforms.Compose([transforms.ToTensor()])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return "Error processing the image. Please make sure it's a valid image file."

    try:
        img = transform_val(img).unsqueeze(0).to(device)
    except Exception as e:
        return "Error processing the image. Please try again."

    # Make predictions
    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output, dim=-1)
        predicted_class_index = torch.argmax(probabilities, dim=-1).item()

    # Decode predictions
    class_label = classes[predicted_class_index]
    return class_label

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        imagefile = request.files['imagefile']
        image_path = os.path.join("images", imagefile.filename)

        try:
            imagefile.save(image_path)
        except Exception as e:
            return render_template('index.html', error="Error saving the image. Please try again.")

        try:
            predicted_class = predict_tumor_class(image_path)
        except Exception as e:
            return render_template('index.html', error="Error predicting tumor class. Please try again.")

        return render_template('index.html', prediction=predicted_class)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)
