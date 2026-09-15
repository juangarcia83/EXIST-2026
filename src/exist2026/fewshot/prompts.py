"""Prompt assembly for the few-shot cascade.

All prompt *text* lives under ``prompts/`` as plain files — the long system
prompts as ``.txt``, the short user turns and the optional rationales as
``.yaml``. Code never embeds prompt strings: the published prompts and the ones
a run actually sends are the same bytes, which is the only way a reader can
check the paper's claims against the repository.

The message list produced here is the "Qwen style" chat format (``content`` as
a list of typed blocks). Backends that want something else normalize it; see
:mod:`exist2026.fewshot.backends`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from exist2026.config import PROMPT_DIR
from exist2026.taxonomy import Modality, Subtask, subtask_key

#: Filename stem of each subtask's system prompt.
PROMPT_STEMS = {
    Subtask.IDENTIFICATION: "sexism_identification",
    Subtask.INTENTION: "source_intention",
    Subtask.CATEGORIZATION: "sexism_categorization",
}


def prompt_path(modality: Modality, subtask: Subtask, root: Path | None = None) -> Path:
    root = root or PROMPT_DIR
    name = f"T{modality.task_number}.{subtask.index}_{PROMPT_STEMS[subtask]}.txt"
    return root / modality.value / name


@cache
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


@cache
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def system_prompt(modality: Modality, subtask: Subtask, root: Path | None = None) -> str:
    """The verbatim system prompt of one subtask."""
    return _read_text(prompt_path(modality, subtask, root))


def questions(modality: Modality, root: Path | None = None) -> dict[str, Any]:
    return _read_yaml((root or PROMPT_DIR) / modality.value / "questions.yaml")


def rationales(modality: Modality, root: Path | None = None) -> dict[str, Any]:
    return _read_yaml((root or PROMPT_DIR) / modality.value / "rationales.yaml")


@dataclass(frozen=True)
class MediaSpec:
    """How to reference the instance's media inside a message block.

    Memes are plain images. Videos carry sampling hints that only some backends
    honour (Qwen-VL reads ``fps``/``max_pixels``; Gemma's processor samples on
    its own), so they are attached and ignored where irrelevant.
    """

    kind: str  # "image" | "video"
    fps: float | None = None
    max_pixels: int | None = None

    @classmethod
    def for_modality(
        cls, modality: Modality, fps: float | None = None, max_pixels: int | None = None
    ) -> MediaSpec:
        if modality is Modality.MEMES:
            return cls(kind="image")
        return cls(kind="video", fps=fps, max_pixels=max_pixels)

    def block(self, path: str) -> dict[str, Any]:
        block: dict[str, Any] = {"type": self.kind, self.kind: str(path)}
        if self.fps is not None:
            block["fps"] = self.fps
        if self.max_pixels is not None:
            block["max_pixels"] = self.max_pixels
        return block


class PromptBuilder:
    """Turns a pool plus a query instance into a chat-template message list.

    One builder serves one subtask. It resolves exemplar media and text through
    the callables it is given, so it stays independent of how the corpus is
    stored.
    """

    def __init__(
        self,
        modality: Modality,
        subtask: Subtask,
        media: MediaSpec,
        *,
        use_rationale: bool = False,
        prompt_root: Path | None = None,
    ) -> None:
        self.modality = modality
        self.subtask = subtask
        self.media = media
        self.use_rationale = use_rationale
        self.key = subtask_key(modality, subtask)
        self.system = system_prompt(modality, subtask, prompt_root)
        block = questions(modality, prompt_root)
        self.question: str = block[self.key]
        self._with_text: str = block["with_text"]
        self._without_text: str = block["without_text"]
        self._rationales = rationales(modality, prompt_root).get(self.key, {}) if use_rationale else {}
        if use_rationale and not self._rationales:
            raise ValueError(
                f"use_rationale is on but prompts/{modality.value}/rationales.yaml "
                f"has no entry for {self.key}"
            )

    # -- turns ------------------------------------------------------------- #

    def user_turn(self, media_path: str, text: str) -> dict[str, Any]:
        """The media block first, then the text — the order Gemma's guide asks
        for and Qwen-VL accepts without penalty."""
        text = (text or "").strip()
        preamble = self._with_text.format(text=text) if text else self._without_text
        return {
            "role": "user",
            "content": [
                self.media.block(media_path),
                {"type": "text", "text": f"{preamble}\n{self.question}"},
            ],
        }

    def answer_turn(self, exemplar: Mapping[str, Any]) -> dict[str, Any]:
        return {"role": "assistant", "content": self.format_answer(exemplar)}

    def format_answer(self, exemplar: Mapping[str, Any]) -> str:
        """The gold answer of one exemplar, in the exact output contract."""
        if self.subtask is Subtask.CATEGORIZATION:
            labels = list(exemplar["labels"])
            answer = ", ".join(labels)
            primary = labels[0] if labels else None
        else:
            answer = primary = exemplar["label"]
        if not self.use_rationale:
            return answer
        return f"{answer} - {self._rationale_for(primary)}"

    def _rationale_for(self, label: str | None) -> str:
        rules: Mapping[str, str] = self._rationales.get("rules", {})
        defaults: Mapping[str, str] = self._rationales.get("default_rule", {})
        rule_id = defaults.get(label, label)
        text = rules.get(rule_id)
        if text is None:
            raise KeyError(f"no rationale for {label!r} in prompts/{self.modality.value}")
        return f"{rule_id}: {text}" if rule_id != label else text

    # -- full prompt -------------------------------------------------------- #

    def build(
        self,
        query_media: str,
        query_text: str,
        pool: Sequence[Mapping[str, Any]],
        resolve_media,
        resolve_text,
    ) -> list[dict[str, Any]]:
        """``[system] + [user, assistant] * len(pool) + [user]``.

        The trailing user turn has no answer: ``apply_chat_template(...,
        add_generation_prompt=True)`` closes it so the model continues.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        for exemplar in pool:
            instance_id = str(exemplar["id"])
            messages.append(self.user_turn(resolve_media(instance_id), resolve_text(instance_id)))
            messages.append(self.answer_turn(exemplar))
        messages.append(self.user_turn(query_media, query_text))
        return messages


def builders_for(
    modality: Modality,
    *,
    use_rationale: bool = False,
    fps: float | None = None,
    max_pixels: int | None = None,
    prompt_root: Path | None = None,
) -> dict[str, PromptBuilder]:
    """One :class:`PromptBuilder` per subtask, keyed by subtask key."""
    media = MediaSpec.for_modality(modality, fps=fps, max_pixels=max_pixels)
    return {
        subtask_key(modality, subtask): PromptBuilder(
            modality, subtask, media, use_rationale=use_rationale, prompt_root=prompt_root
        )
        for subtask in Subtask
    }
