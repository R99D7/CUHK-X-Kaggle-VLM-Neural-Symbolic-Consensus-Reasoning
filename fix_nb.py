import json

with open('cuhk-moondream-gpu.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_nb = {
  "cells": [
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "%%writefile run_inference.py\n",
        "import os\n",
        "import gc\n",
        "import json\n",
        "import torch\n",
        "from PIL import Image\n",
        "import pandas as pd\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "from transformers.modeling_utils import PreTrainedModel\n",
        "PreTrainedModel.all_tied_weights_keys = property(lambda self: {})\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "print(\"Loading Moondream2 directly from HuggingFace with updated transformers...\")\n",
        "MODEL_ID = \"vikhyatk/moondream2\"\n",
        "try:\n",
        "    model = AutoModelForCausalLM.from_pretrained(\n",
        "        MODEL_ID, trust_remote_code=True\n",
        "    ).to(\"cuda\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n",
        "except Exception as e:\n",
        "    print(\"Failed to load model:\", e)\n",
        "    raise\n",
        "\n",
        "test_df = pd.read_csv('/kaggle/input/cuhk-sysu-llm-competition/test_qa.csv')\n",
        "video_dir = '/kaggle/input/cuhk-sysu-llm-competition/test_video'\n",
        "\n",
        "target_df = test_df[test_df['category'].isin(['multi', 'sequence'])]\n",
        "print(f\"Evaluating {len(target_df)} questions (multi and sequence)...\")\n",
        "\n",
        "final_preds = []\n",
        "for idx, row in target_df.iterrows():\n",
        "    video_path = os.path.join(video_dir, str(row['video_id']) + '.mp4')\n",
        "    if not os.path.exists(video_path):\n",
        "        final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})\n",
        "        continue\n",
        "        \n",
        "    try:\n",
        "        import cv2\n",
        "        cap = cv2.VideoCapture(video_path)\n",
        "        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n",
        "        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)\n",
        "        ret, frame = cap.read()\n",
        "        cap.release()\n",
        "        \n",
        "        if not ret:\n",
        "            final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})\n",
        "            continue\n",
        "            \n",
        "        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
        "        image = Image.fromarray(frame_rgb)\n",
        "        \n",
        "        question_prompt = (\n",
        "            f\"Question: {row['question']}\\n\"\n",
        "            f\"A) {row['a']}\\nB) {row['b']}\\nC) {row['c']}\\nD) {row['d']}\\n\"\n",
        "            f\"Category: {row['category']}\\n\"\n",
        "            \"Based on the image, strictly output ONLY the correct option letter(s) (A, B, C, or D).\"\n",
        "        )\n",
        "        \n",
        "        enc_image = model.encode_image(image)\n",
        "        answer = model.answer_question(enc_image, question_prompt, tokenizer)\n",
        "        \n",
        "        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])\n",
        "        if len(pred) == 0: pred = 'A'\n",
        "        \n",
        "        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})\n",
        "    except Exception as e:\n",
        "        print(f\"Error on {row['qa_id']}: {e}\")\n",
        "        final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})\n",
        "        \n",
        "pd.DataFrame(final_preds).to_csv('submission_moondream_gpu.csv', index=False)\n",
        "print(\"Finished creating submission_moondream_gpu.csv\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip install einops > /dev/null\n",
        "!python run_inference.py\n"
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

with open('cuhk-moondream-gpu.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=2)

print("Notebook updated successfully.")
