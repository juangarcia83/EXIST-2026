"""Label space of the EXIST 2026 shared task.

Tasks 2 (memes) and 3 (TikTok videos) share the same three-subtask structure, so
everything here is expressed once and addressed through a :class:`Modality`.

Subtask keys follow the ``t<task><subtask>`` convention used by the official
submission format: ``t21``/``t22``/``t23`` for memes and ``t31``/``t32``/``t33``
for videos.
"""

from __future__ import annotations

from enum import Enum

#: Sexism categories of subtask x.3, in the canonical order used by every
#: multi-hot / soft-label vector in this package.
SEXISM_CATEGORIES: tuple[str, ...] = (
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
)

#: Label of an annotation that must always be discarded before aggregating.
UNKNOWN = "UNKNOWN"

#: In subtask x.2 the organizers encode "not sexist" as ``"-"``.
INTENTION_NO_RAW = "-"

IDENTIFICATION_LABELS: tuple[str, ...] = ("NO", "YES")
INTENTION_LABELS: tuple[str, ...] = ("NO", "DIRECT", "JUDGEMENTAL")

IDENTIFICATION_INT_TO_STR = dict(enumerate(IDENTIFICATION_LABELS))
INTENTION_INT_TO_STR = dict(enumerate(INTENTION_LABELS))
INTENTION_STR_TO_INT = {v: k for k, v in INTENTION_INT_TO_STR.items()}


class Modality(str, Enum):
    """The two EXIST 2026 modalities this repository covers."""

    MEMES = "memes"
    VIDEOS = "videos"

    @property
    def task_number(self) -> int:
        """Official task number: 2 for memes, 3 for videos."""
        return 2 if self is Modality.MEMES else 3

    @property
    def media_field(self) -> str:
        """Key holding the media filename inside a dataset record."""
        return "meme" if self is Modality.MEMES else "video"

    @property
    def media_suffix(self) -> str:
        return ".jpeg" if self is Modality.MEMES else ".mp4"


class Subtask(str, Enum):
    """The three subtasks, independent of the modality."""

    IDENTIFICATION = "identification"
    INTENTION = "intention"
    CATEGORIZATION = "categorization"

    @property
    def index(self) -> int:
        """1-based position inside the hierarchy (x.1, x.2, x.3)."""
        return [Subtask.IDENTIFICATION, Subtask.INTENTION, Subtask.CATEGORIZATION].index(self) + 1


#: Number of output units a soft-label head needs per subtask.
SOFT_LABEL_WIDTH = {
    Subtask.IDENTIFICATION: 2,
    Subtask.INTENTION: 3,
    Subtask.CATEGORIZATION: 1 + len(SEXISM_CATEGORIES),
}


def subtask_key(modality: Modality, subtask: Subtask) -> str:
    """``t21`` … ``t33``, the key used for predictions, caches and filenames."""
    return f"t{modality.task_number}{subtask.index}"


def subtask_keys(modality: Modality) -> tuple[str, str, str]:
    return tuple(subtask_key(modality, s) for s in Subtask)  # type: ignore[return-value]


def parse_subtask_key(key: str) -> tuple[Modality, Subtask]:
    """Inverse of :func:`subtask_key`. Raises ``ValueError`` on unknown keys."""
    if len(key) != 3 or key[0] != "t" or key[1] not in "23" or key[2] not in "123":
        raise ValueError(f"not a subtask key: {key!r}")
    modality = Modality.MEMES if key[1] == "2" else Modality.VIDEOS
    subtask = list(Subtask)[int(key[2]) - 1]
    return modality, subtask


def subtask_of(key: str) -> Subtask:
    return parse_subtask_key(key)[1]


def annotation_field(modality: Modality, subtask: Subtask) -> str:
    """Name of the raw-annotation list inside a dataset record."""
    return f"labels_task{modality.task_number}_{subtask.index}"


def official_submission_stem(modality: Modality, subtask: Subtask) -> str:
    """``task2_1`` … ``task3_3`` as required by the submission guidelines."""
    return f"task{modality.task_number}_{subtask.index}"
