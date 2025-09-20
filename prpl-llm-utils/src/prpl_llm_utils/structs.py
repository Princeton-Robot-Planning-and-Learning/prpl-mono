"""Data structures."""

from dataclasses import dataclass
from typing import Any, Hashable

import imagehash
import PIL.Image

from prpl_llm_utils.utils import consistent_hash


@dataclass(frozen=True)
class Query:
    """A query for a pretrained large model."""

    prompt: str
    imgs: list[PIL.Image.Image] | None = None
    hyperparameters: dict[str, Hashable] | None = None

    def get_id(self) -> Hashable:
        """Get a unique and hashable ID for this query."""
        # Hash the images first, since that requires a special library.
        img_hash_list: list[str] = []
        if self.imgs is not None:
            for img in self.imgs:
                img_hash_list.append(str(imagehash.phash(img)))
        entries: list[Hashable] = [self.prompt, tuple(img_hash_list)]
        if self.hyperparameters:
            for key in sorted(self.hyperparameters):
                entries.append((key, self.hyperparameters[key]))
        return tuple(entries)

    def get_readable_id(self) -> str:
        """Get an ID that is at least somewhat human readable."""
        prompt_prefix = self.prompt[:32]
        unique_id = str(hash(self))
        return f"{prompt_prefix}__{unique_id}"

    def __hash__(self) -> int:
        """Consistent hashing between runs."""
        return consistent_hash(self.get_id())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Query):
            return False
        return self.get_id() == other.get_id()


@dataclass(frozen=True)
class Response:
    """A response from a pretrained large model."""

    text: str
    metadata: dict[str, Any]  # number tokens, etc.
