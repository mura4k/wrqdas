import torch
import os
from BudaOCR.Modules import EasterNetwork
from BudaOCR.Modules import WylieEncoder
from BudaOCR.Utils import create_dir

# === CONFIG ===
checkpoint = "OCRModel.pth"
output_dir = "onnx_output"
onnx_file = "OCRModel.onnx"
input_width = 3200
input_height = 100
charset = [
    "!", "#", "%", "'", "°", "^", "$", "`", "(", ")", "+", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", "=", "?", "@",
    "A", "D", "H", "I", "M", "N", "R", "S", "T", "U", "W", "X", "Y",
    "[", "\\", "]", "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "w", "y", "x", "v",
    "z", "|", "~", "§", "&", "ä", "ü", "ö", "<", ">", ";", "}", "卍", "卐"
]

# === Encoder and Model ===
encoder = WylieEncoder(charset)
num_classes = encoder.num_classes()

network = EasterNetwork(
    num_classes=num_classes,
    image_width=input_width,
    image_height=input_height,
    mean_pooling=True
)

# === Load weights ===
network.load_model(checkpoint)

# === Export to ONNX ===
create_dir(output_dir)
network.export_onnx(out_dir=output_dir, model_name="OCRModel", opset=17)

print(f"✅ Model exported to: {output_dir}/{onnx_file}")
