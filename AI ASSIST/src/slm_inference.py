"""
BFSI Call Center AI - Small Language Model (Tier 2)
Used ONLY when no strong dataset similarity match exists.
Runs locally on modest hardware.
"""

from pathlib import Path
from typing import Optional


class SLMInference:
    """
    Tier 2: Local fine-tuned SLM.
    Generates response when no dataset match.
    Uses TinyLlama or fine-tuned weights.
    """

    def __init__(self, config: dict, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.weights_path = self.base_path / config.get("weights_path", "models/slm_weights")
        self.max_new_tokens = int(config.get("max_new_tokens", 256))
        self.temperature = float(config.get("temperature", 0.3))
        self.use_finetuned = config.get("use_finetuned", True)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required. pip install transformers torch")

        # Prefer fine-tuned weights if available
        model_path = str(self.weights_path) if self.weights_path.exists() and self.use_finetuned else self.model_name
        tokenizer_path = model_path

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto" if self._has_cuda() else None,
            )
            if self._model.device.type == "cpu":
                self._model = self._model.float()
        except Exception:
            # Fallback to base model if fine-tuned not found
            if model_path != self.model_name:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto" if self._has_cuda() else None,
                )
            else:
                raise
        return self._model, self._tokenizer

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _format_prompt(self, query: str) -> str:
        """Format query for instruction-following model."""
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a professional BFSI call center assistant. Respond helpfully, accurately, and in a compliant manner. Do NOT guess financial numbers, interest rates, or policy details. If unsure, direct the customer to official channels.

### Input:
{query}

### Response:
"""

    def generate(self, query: str) -> str:
        """
        Generate response using local SLM.
        Tier 2: Called only when no dataset match.
        """
        model, tokenizer = self._load_model()
        prompt = self._format_prompt(query)
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device") and self._model.device.type != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response after "### Response:"
        if "### Response:" in full_text:
            response = full_text.split("### Response:")[-1].strip()
        else:
            response = full_text[len(prompt):].strip()
        return response
