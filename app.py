# Import and class names setup
import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open('class_names.txt', 'r') as f:
  class_names= [name.strip() for name in f.readlines()]


# Model and transforms preparation
effnet_model, effnet_transform= create_model()
# Load state dict
effnet_model.load_state_dict(torch.load(
    f= 'pretrained_effnetb0_feature_extractor_brain_tumor.pth',
    map_location= torch.device('cpu')
    )
)

# Predict function

def predict(img)-> Tuple[Dict, float]:
  # start a timer
  start_time= timer()

  #transform the input image for use with effnet b2
  transform_image= effnet_transform(img).unsqueeze(0)

  #put model into eval mode, make pred
  effnet_model.eval()
  with torch.inference_mode():
    pred_logits= effnet_model(transform_image)
    pred_prob= torch.softmax(pred_logits, dim=1)

  # create a pred label and pred prob dict
  pred_label_and_prob= {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))}


  # calc pred time
  stop_time= timer()
  pred_time= round(stop_time - start_time, 4)


  # return pred dict and pred time
  return pred_label_and_prob, pred_time

# create gradio app
title= 'Brain Tumor Prediction App'
description= 'An EfficientnetB0 feature extractor Computer vision model to classify if an brain xray image have brain tumor and the type of tumor that is in the brain'
article= 'Created at [To be uploaded].'

# Create the gradio demo
demo= gr.Interface(fn= predict,
                   inputs=gr.Image(type='pil'),
                   outputs= [gr.Label(num_top_classes=5, label= 'predictions'),
                             gr.Number(label= 'Prediction time (S)')],
                   title= title,
                   description= description,
                   article= article
                   )

# Launch the demo
demo.launch()
