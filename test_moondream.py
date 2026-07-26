import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

print("Loading Moondream2 model...")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Create dummy image
img_array = np.zeros((224, 224, 3), dtype=np.uint8)
image = Image.fromarray(img_array)

print("Encoding image...")
start_time = time.time()
enc_image = model.encode_image(image)
print(f"Image encode time: {time.time() - start_time:.2f}s")

print("Generating answer...")
start_time = time.time()
answer = model.answer_question(enc_image, "What is in this image?", tokenizer)
print(answer)
print(f"Answer generation time: {time.time() - start_time:.2f}s")
