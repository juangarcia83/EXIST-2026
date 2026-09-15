"""Vision-language backends for the few-shot cascade.

Two backend families are supported and are interchangeable: the pipeline,
prompts and pools above this module never branch on which one is loaded. The
differences live here and are only about input preparation — Qwen-VL extracts
frames through ``qwen_vl_utils`` and takes ``images=``/``videos=`` tensors,
while Gemma's processor tokenizes the chat template directly.

``torch`` and ``transformers`` are imported inside the functions that need
them, so importing :mod:`exist2026` on a machine without a GPU stack (CI, a
laptop preparing configs) stays cheap and side-effect free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from exist2026.config import BackendSpec, GenerationSpec


class BackendLoadError(RuntimeError):
    """No candidate model class could be loaded for a backend."""


def _torch():
    import torch

    return torch


def use_four_bit(threshold_bytes: float) -> bool:
    """Whether to quantize, based on the VRAM actually available."""
    torch = _torch()
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory < threshold_bytes


def _load_kwargs(four_bit: bool) -> dict[str, Any]:
    import torch
    from transformers import BitsAndBytesConfig

    if not four_bit:
        # transformers >= 4.57 renamed torch_dtype to dtype.
        return {"device_map": "auto", "dtype": torch.bfloat16}
    return {
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
    }


class VlmBackend(ABC):
    """A loaded model plus everything needed to answer one prompt."""

    def __init__(self, spec: BackendSpec, generation: GenerationSpec, model, processor) -> None:
        self.spec = spec
        self.generation = generation
        self.model = model
        self.processor = processor

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @abstractmethod
    def prepare(self, messages: Sequence[Mapping[str, Any]]):
        """Model inputs for one message list, already on the model's device."""

    def generate(self, messages: Sequence[Mapping[str, Any]], max_new_tokens: int) -> str:
        """Decode the continuation of ``messages``, without the prompt."""
        torch = _torch()
        inputs = self.prepare(messages)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.generation.do_sample,
                temperature=None,
                top_p=None,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def prompt_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        """Prompt length in tokens, media included — used by the budget check."""
        return int(self.prepare(messages)["input_ids"].shape[1])

    def max_context(self, practical_cap: int = 262_144) -> int:
        """Effective context window; some tokenizers report a nonsense maximum."""
        config_max = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        tokenizer_max = getattr(tokenizer, "model_max_length", None) if tokenizer else None
        value = config_max or tokenizer_max or 32_768
        return practical_cap if value > 1_000_000 else int(value)


class QwenBackend(VlmBackend):
    """Qwen3.5 / Qwen-VL: frames come from ``qwen_vl_utils``."""

    def prepare(self, messages):
        from qwen_vl_utils import process_vision_info

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Older transformers releases do not accept enable_thinking.
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        return self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)


class GemmaBackend(VlmBackend):
    """Gemma-4: the processor tokenizes the chat template itself."""

    def prepare(self, messages):
        return self.processor.apply_chat_template(
            normalize_for_gemma(messages),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)


def normalize_for_gemma(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adapt the canonical message format to what Gemma's template expects.

    Two changes: every ``content`` becomes a list of blocks (system and
    assistant turns included), and image paths are opened as PIL objects, which
    sidesteps the processor's handling of local paths. Video blocks keep their
    path — the processor reads the file itself.
    """
    from PIL import Image

    normalized = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = []
            for block in content:
                if block.get("type") == "image":
                    source = block.get("image") or block.get("url") or block.get("path")
                    blocks.append({"type": "image", "image": Image.open(str(source)).convert("RGB")})
                else:
                    blocks.append(dict(block))
        normalized.append({"role": message["role"], "content": blocks})
    return normalized


def load_backend(spec: BackendSpec, generation: GenerationSpec, *, verbose: bool = True) -> VlmBackend:
    """Load the first candidate model class that works.

    The concrete class name for a recent VLM moves between transformers
    releases, so a backend declares several and the first that loads wins. The
    error raised when none does names the two usual causes (a stale
    transformers, or an unaccepted model licence).
    """
    import transformers
    from transformers import AutoProcessor

    four_bit = use_four_bit(spec.four_bit_below_vram)
    kwargs = _load_kwargs(four_bit)
    failures = []
    for class_name in spec.model_classes:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            failures.append(f"{class_name}: not exposed by transformers {transformers.__version__}")
            continue
        try:
            model = model_class.from_pretrained(spec.model_id, **kwargs)
            processor = AutoProcessor.from_pretrained(spec.model_id)
        except Exception as error:
            failures.append(f"{class_name}: {type(error).__name__}: {error}")
            continue
        model.eval()
        if verbose:
            print(f"loaded {spec.model_id} via {class_name} (4-bit={four_bit})")
        backend_class = QwenBackend if spec.needs_qwen_vl_utils else GemmaBackend
        return backend_class(spec, generation, model, processor)

    detail = "\n  ".join(failures)
    raise BackendLoadError(
        f"could not load backend {spec.name!r} ({spec.model_id}).\n  {detail}\n"
        "Check that transformers >= 4.57 is installed and that you have accepted "
        "the model licence on the Hugging Face Hub."
    )
