"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys

import torch.onnx
from model import EAST

origin_model_path="./saved_model/mb3_512_model_epoch_535.pth"

model = EAST().to("cuda")
model.load_state_dict(torch.load(origin_model_path))
model.eval()

model_path = "model/mbv3_512_east.onnx"

dummy_input = torch.randn(1, 3, 512, 512).to("cuda")

torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['east_detect'])
