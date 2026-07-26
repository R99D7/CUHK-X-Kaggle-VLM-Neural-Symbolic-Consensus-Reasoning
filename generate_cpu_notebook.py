import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# CUHK Video QA: Moondream2 CPU Inference"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q transformers decord einops"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from PIL import Image\n",
                "from decord import VideoReader, cpu\n",
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
                "import time\n\n",
                "# Optimize CPU execution\n",
                "torch.set_num_threads(4)\n\n",
                "print(\"Loading Moondream2 on CPU...\")\n",
                "model_id = \"vikhyatk/moondream2\"\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    model_id, trust_remote_code=True, torch_dtype=torch.float32\n",
                ")\n",
                "model.eval()\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_id)\n\n",
                "test_df = pd.read_csv('/kaggle/input/cuhk-x-competition-large-model-track/test_qa.csv')\n",
                "predictions = []\n\n",
                "start_time = time.time()\n\n",
                "for idx, row in test_df.iterrows():\n",
                "    qa_id = row['qa_id']\n",
                "    video_path = f\"/kaggle/input/cuhk-x-competition-large-model-track/video/{row['video_id']}.mp4\"\n",
                "    \n",
                "    try:\n",
                "        vr = VideoReader(video_path, ctx=cpu(0))\n",
                "        total_frames = len(vr)\n",
                "        frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)\n",
                "        frames = vr.get_batch(frame_indices).asnumpy()\n",
                "        pil_frames = [Image.fromarray(f).convert(\"RGB\") for f in frames]\n",
                "    except:\n",
                "        pil_frames = [Image.new('RGB', (224, 224))] * 4\n",
                "        \n",
                "    w, h = pil_frames[0].size\n",
                "    grid = Image.new('RGB', (w*2, h*2))\n",
                "    grid.paste(pil_frames[0], (0, 0))\n",
                "    grid.paste(pil_frames[1], (w, 0))\n",
                "    grid.paste(pil_frames[2], (0, h))\n",
                "    grid.paste(pil_frames[3], (w, h))\n",
                "    \n",
                "    question = row['question']\n",
                "    A, B, C, D = row['A'], row['B'], row['C'], row['D']\n",
                "    \n",
                "    if row['category'] == 'multi':\n",
                "        prompt = (\n",
                "            f\"Question: {question}\\n\"\n",
                "            f\"A: {A}\\n\"\n",
                "            f\"B: {B}\\n\"\n",
                "            f\"C: {C}\\n\"\n",
                "            f\"D: {D}\\n\"\n",
                "            \"This is a multiple-choice question where MULTIPLE answers may be correct. \"\n",
                "            \"List ALL correct letters combined (e.g. AB, ACD). ONLY output the letters, nothing else.\"\n",
                "        )\n",
                "    else:\n",
                "        prompt = (\n",
                "            f\"Question: {question}\\n\"\n",
                "            f\"A: {A}\\n\"\n",
                "            f\"B: {B}\\n\"\n",
                "            f\"C: {C}\\n\"\n",
                "            f\"D: {D}\\n\"\n",
                "            \"Based on the images, answer the question with ONLY the correct letter (A, B, C, or D). Do not explain.\"\n",
                "        )\n",
                "        \n",
                "    with torch.no_grad():\n",
                "        enc_image = model.encode_image(grid)\n",
                "        answer = model.answer_question(enc_image, prompt, tokenizer)\n",
                "        \n",
                "    ans = \"\"\n",
                "    for letter in ['A', 'B', 'C', 'D']:\n",
                "        if letter in answer.upper():\n",
                "            ans += letter\n",
                "    if not ans:\n",
                "        ans = \"A\"\n",
                "        \n",
                "    predictions.append(ans)\n",
                "    \n",
                "    if (idx + 1) % 50 == 0:\n",
                "        elapsed = time.time() - start_time\n",
                "        print(f\"Processed {idx + 1}/{len(test_df)} videos. Elapsed: {elapsed:.2f}s\")\n\n",
                "test_df['prediction'] = predictions\n",
                "sub = test_df[['qa_id', 'prediction']]\n",
                "sub.to_csv('submission.csv', index=False)\n",
                "print(\"Finished processing all videos. Saved submission.csv\")\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("cuhk-moondream-cpu.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
