import torch
import torchvision
from torch import nn

def create_model(num_classes:int=4,
                 seed:int=42):
  # Create Effnet pretrained model
  weights= torchvision.models.EfficientNet_B0_Weights.DEFAULT
  transforms= weights.transforms()
  model= torchvision.models.efficientnet_b0(weights=weights)

  # Freeze all layers in the base model
  for param in model.parameters():
    param.requires_grad= False

  # Change the classifier layer
  torch.manual_seed(seed)
  model.classifier= nn.Sequential(
    nn.Dropout(p=0.2, inplace= True),
    nn.Linear(in_features= 1280, out_features= num_classes)
    )
  return model, transforms
