import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained('moondream2', trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('moondream2', local_files_only=True)

img = Image.new('RGB', (378, 378), color='blue')
enc = model.encode_image(img)
prompt = """Question: What color is this image?
A) Red
B) Blue
C) Green
D) Yellow
Select the correct option. Output ONLY the letter (A, B, C, or D) and nothing else."""
print('Answer:', model.answer_question(enc, prompt, tokenizer))
