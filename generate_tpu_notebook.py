import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# CUHK Video QA: TPU Partial Fine-Tuning with Hard Example Mining"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q transformers decord"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile train.py\n",
                "import os\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from PIL import Image\n",
                "from decord import VideoReader, cpu\n",
                "from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, TrainingArguments, Trainer\n",
                "from torch.utils.data import Dataset\n",
                "import torch_xla.core.xla_model as xm\n",
                "import torch_xla.distributed.xla_multiprocessing as xmp\n\n",
                "def get_dataset():\n",
                "    train_df = pd.read_csv('/kaggle/input/cuhk-x-competition-large-model-track/training_qa.csv')\n",
                "    train_df = train_df[train_df['category'] == 'multi'].reset_index(drop=True)\n",
                "    return train_df\n\n",
                "class CUHKVideoDataset(Dataset):\n",
                "    def __init__(self, df, processor):\n",
                "        self.df = df\n",
                "        self.processor = processor\n",
                "        self.processor.tokenizer.padding_side = 'right'\n",
                "        if self.processor.tokenizer.pad_token is None:\n",
                "            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token\n\n",
                "    def __len__(self):\n",
                "        return len(self.df)\n\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        video_path = f\"/kaggle/input/cuhk-x-competition-large-model-track/video/{row['video_id']}.mp4\"\n",
                "        try:\n",
                "            vr = VideoReader(video_path, ctx=cpu(0))\n",
                "            total_frames = len(vr)\n",
                "            frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)\n",
                "            frames = vr.get_batch(frame_indices).asnumpy()\n",
                "            frames = [Image.fromarray(f) for f in frames]\n",
                "        except:\n",
                "            frames = [Image.new('RGB', (224, 224))] * 4\n\n",
                "        question = row['question']\n",
                "        opts = f\"A: {row['A']}\\nB: {row['B']}\\nC: {row['C']}\\nD: {row['D']}\"\n",
                "        prompt = f\"USER: <video>\\nQuestion: {question}\\nOptions:\\n{opts}\\nAnswer exactly with the correct option letter(s).\\nASSISTANT: {row['answer']}\"\n\n",
                "        inputs = self.processor(text=prompt, videos=frames, return_tensors=\"pt\", padding=\"max_length\", max_length=128, truncation=True)\n",
                "        inputs[\"labels\"] = inputs[\"input_ids\"].clone()\n",
                "        inputs[\"labels\"][inputs[\"labels\"] == self.processor.tokenizer.pad_token_id] = -100\n",
                "        return {k: v.squeeze(0) for k, v in inputs.items()}\n\n",
                "def train_model(index):\n",
                "    model_id = 'LanguageBind/Video-LLaVA-7B-hf'\n",
                "    processor = VideoLlavaProcessor.from_pretrained(model_id)\n",
                "    model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)\n\n",
                "    for param in model.video_tower.parameters(): param.requires_grad = False\n",
                "    for param in model.multi_modal_projector.parameters(): param.requires_grad = False\n",
                "    for i in range(24):\n",
                "        for param in model.language_model.model.layers[i].parameters():\n",
                "            param.requires_grad = False\n\n",
                "    train_df = get_dataset()\n",
                "    train_dataset = CUHKVideoDataset(train_df, processor)\n\n",
                "    args = TrainingArguments(\n",
                "        output_dir=\"./videollava_tpu_checkpoints\",\n",
                "        per_device_train_batch_size=1,\n",
                "        gradient_accumulation_steps=2,\n",
                "        learning_rate=2e-5,\n",
                "        num_train_epochs=2,\n",
                "        logging_steps=10,\n",
                "        save_strategy=\"epoch\",\n",
                "        bf16=True,\n",
                "        optim=\"adafactor\",\n",
                "        report_to=\"none\"\n",
                "    )\n\n",
                "    trainer = Trainer(\n",
                "        model=model,\n",
                "        args=args,\n",
                "        train_dataset=train_dataset\n",
                "    )\n\n",
                "    trainer.train()\n",
                "    if index == 0:\n",
                "        trainer.save_model(\"./videollava_tpu_final\")\n",
                "        processor.save_pretrained(\"./videollava_tpu_final\")\n\n",
                "if __name__ == '__main__':\n",
                "    xmp.spawn(train_model, args=(), nprocs=8, start_method='fork')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python train.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile inference.py\n",
                "import os\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from PIL import Image\n",
                "from decord import VideoReader, cpu\n",
                "from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor\n",
                "import torch_xla.core.xla_model as xm\n\n",
                "def main():\n",
                "    if not os.path.exists('./videollava_tpu_final'):\n",
                "        print(\"Training failed or model was not saved.\")\n",
                "        return\n",
                "    device = xm.xla_device()\n",
                "    processor = VideoLlavaProcessor.from_pretrained(\"./videollava_tpu_final\")\n",
                "    model = VideoLlavaForConditionalGeneration.from_pretrained(\"./videollava_tpu_final\", torch_dtype=torch.bfloat16)\n",
                "    model.to(device)\n",
                "    model.eval()\n\n",
                "    test_df = pd.read_csv('/kaggle/input/cuhk-x-competition-large-model-track/test_qa.csv')\n",
                "    predictions = []\n\n",
                "    for idx, row in test_df.iterrows():\n",
                "        video_path = f\"/kaggle/input/cuhk-x-competition-large-model-track/video/{row['video_id']}.mp4\"\n",
                "        try:\n",
                "            vr = VideoReader(video_path, ctx=cpu(0))\n",
                "            total_frames = len(vr)\n",
                "            frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)\n",
                "            frames = vr.get_batch(frame_indices).asnumpy()\n",
                "            frames = [Image.fromarray(f) for f in frames]\n",
                "        except:\n",
                "            frames = [Image.new('RGB', (224, 224))] * 4\n\n",
                "        question = row['question']\n",
                "        opts = f\"A: {row['A']}\\nB: {row['B']}\\nC: {row['C']}\\nD: {row['D']}\"\n",
                "        prompt = f\"USER: <video>\\nQuestion: {question}\\nOptions:\\n{opts}\\nAnswer exactly with the correct option letter(s).\\nASSISTANT:\"\n\n",
                "        inputs = processor(text=prompt, videos=frames, return_tensors=\"pt\")\n",
                "        inputs = {k: v.to(device) for k, v in inputs.items()}\n\n",
                "        with torch.no_grad():\n",
                "            out = model.generate(**inputs, max_new_tokens=5, temperature=0.0)\n\n",
                "        pred_text = processor.batch_decode(out, skip_special_tokens=True)[0]\n",
                "        ans = pred_text.split('ASSISTANT:')[-1].strip()\n",
                "        clean_ans = ''.join([c for c in ans if c in 'ABCD'])\n",
                "        if len(clean_ans) == 0: clean_ans = 'A'\n",
                "        predictions.append(clean_ans)\n",
                "        print(f\"{row['qa_id']}: {clean_ans}\")\n\n",
                "    test_df['prediction'] = predictions\n",
                "    sub = test_df[['qa_id', 'prediction']]\n",
                "    sub.to_csv('submission.csv', index=False)\n",
                "    print(\"Saved submission.csv\")\n\n",
                "if __name__ == '__main__':\n",
                "    main()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python inference.py"
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

with open("cuhk-videollava-tpu-hard-mining.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
