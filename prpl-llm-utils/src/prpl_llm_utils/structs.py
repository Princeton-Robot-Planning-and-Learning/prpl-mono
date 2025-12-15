"""Data structures."""

import hashlib
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
        img_hash_list = []
        if self.imgs is not None:
            img_hash_list = self.robust_image_hash_list()
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

    def robust_image_hash_list(self, size=(512, 512)) -> list[str]:
        """Compute a list of robust image hashes with preprocessing and scaling."""

        hashed = []
        assert self.imgs is not None
        for img in self.imgs:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize(size, PIL.Image.Resampling.LANCZOS)

            try:
                ph = imagehash.phash(img)
                dh = imagehash.dhash(img)
                wh = imagehash.whash(img)
                ch = imagehash.colorhash(img)
            except Exception as e:
                raise RuntimeError(f"Failed to compute image hash: {e}") from e

            combined = "|".join(map(str, [ph, dh, wh, ch]))
            hashed.append(str(hashlib.sha256(combined.encode("utf-8")).hexdigest()))

        return hashed

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
