from pytorch_grad_cam import GradCAMPlusPlus as CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.models as models

def load_model(model_path, device):
    # Initialize the model architecture
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=4)

    # Load the state dictionary into the initialized model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

model_path = "best_model.pth"  # Provide the path to your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = load_model(model_path, device)

threshold = 0.3
show_count = 10
show_class_id = 1
cam = CAM(model=best_model,
          target_layers=[
                         best_model.layer3[-1],  # Specify the target layer
                         # best_model.features[22],
                        ],
          use_cuda=device.type == "cuda")  # Adjust the condition for GPU usage
targets = [ClassifierOutputTarget(show_class_id)]

count = 0
for batch in test_loader:
    x, y = batch
    x = x.to(device)
    out = best_model(x)
    pred = out.softmax(dim=-1).detach().cpu().numpy()
    predicted_class = out.argmax(dim=-1)

    for i in range(len(x)):
        if y[i] == show_class_id and y[i] == predicted_class[i] and pred[i, y[i]] > threshold:
            # if tumors & model is predicting it right
            grayscale_cam = cam(input_tensor=x[i:i+1, :], targets=targets)
            visualization = show_cam_on_image(x[i, :].cpu().numpy().transpose(1, 2, 0),
                                              grayscale_cam[0, :],
                                              use_rgb=True)
            plt.imshow(visualization)
            plt.title(test.classes[y[i]] + f' score: {pred[i, y[i]]:.2f}')
            plt.show()
            count += 1
        if count >= show_count:
            break

    if count >= show_count:
        break
