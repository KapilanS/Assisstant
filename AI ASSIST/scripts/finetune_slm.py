"""
BFSI Call Center AI - SLM Fine-Tuning Pipeline
Fine-tunes TinyLlama (or similar) using the Alpaca BFSI dataset.
Model weights saved to models/slm_weights for local inference.
Requires: transformers, datasets, peft, accelerate
"""

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def load_alpaca_dataset(path: Path) -> list:
    """Load Alpaca BFSI dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def format_for_training(alpaca_item: dict) -> str:
    """Convert Alpaca (instruction, input, output) to training text."""
    inst = alpaca_item.get("instruction", "")
    inp = alpaca_item.get("input", "")
    out = alpaca_item.get("output", "")
    if inp:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    return prompt


def main():
    dataset_path = _PROJECT_ROOT / "data" / "alpaca_dataset.json"
    output_dir = _PROJECT_ROOT / "models" / "slm_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run: python scripts/generate_dataset.py")
        return 1

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import Dataset
    except ImportError:
        print("Install: pip install transformers datasets peft accelerate")
        return 1

    data = load_alpaca_dataset(dataset_path)
    texts = [format_for_training(item) for item in data]
    ds = Dataset.from_dict({"text": texts})

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )
        labels = []
        for ids, mask in zip(result["input_ids"], result["attention_mask"]):
            label = [(-100 if m == 0 else i) for i, m in zip(ids, mask)]
            labels.append(label)
        result["labels"] = labels
        return result

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,  # Set True if CUDA available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Fine-tuned model saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
