import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import sys

model_path = "vit-customer-model/final"
processor = AutoImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
inputs = processor(img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    label = model.config.id2label[pred]

print(f"Predicted Label: {label}")
