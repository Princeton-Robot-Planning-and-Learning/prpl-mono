"""Data structure for object-centric state representation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Sequence,
)

import numpy as np
from prpl_utils.utils import consistent_hash
from tabulate import tabulate
from typing_extensions import Self

from relational_structs.common import Array
from relational_structs.objects import Object, Type


@dataclass
class ObjectCentricState:
    """Storage for object features."""

    data: Dict[Object, Array]
    type_features: Dict[Type, List[str]]

    def __post_init__(self) -> None:
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == len(self.type_features[obj.type])

    @cached_property
    def _hash(self) -> int:
        items = []
        for obj in sorted(self.data):
            obj_repr = (obj.name, obj.type.name)
            features = tuple(self.data[obj])
            items.append((obj_repr, features))
        return consistent_hash(tuple(items))

    def __hash__(self) -> int:
        return self._hash

    def __iter__(self) -> Iterator[Object]:
        """An iterator over the state's objects, in sorted order."""
        return iter(sorted(self.data))

    def __getitem__(self, key: Object) -> Array:
        return self.data[key]

    def __eq__(self, other: Any) -> bool:
        """NOTE: it is better to call allclose() to control precision."""
        if not isinstance(other, ObjectCentricState):
            return False
        return self.allclose(other)

    def get(self, obj: Object, feature_name: str) -> Any:
        """Look up an object feature by name."""
        idx = self.type_features[obj.type].index(feature_name)
        return self.data[obj][idx]

    def set(self, obj: Object, feature_name: str, feature_val: Any) -> None:
        """Set the value of an object feature by name."""
        idx = self.type_features[obj.type].index(feature_name)
        self.data[obj][idx] = feature_val

    def get_objects(self, object_type: Type) -> List[Object]:
        """Return objects of the given type in the order of __iter__()."""
        return [o for o in self if o.is_instance(object_type)]

    def get_object_from_name(self, name: str) -> Object:
        """Look up an object from its name."""
        matches = [o for o in self if o.name == name]
        if not matches:
            raise ValueError(f"Object '{name}' not found in state")
        return matches[0]

    def vec(self, objects: Sequence[Object],
            dtype: Any = np.float32
            ) -> Array:
        """Concatenated vector of features for each of the objects in the given
        ordered list."""
        feats: List[Array] = []
        if len(objects) == 0:
            return np.zeros(0, dtype=dtype)
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats).astype(dtype)

    @classmethod
    def from_vec(
        cls,
        vec: Array,
        constant_objects: list[Object],
        type_features: Dict[Type, List[str]],
    ):
        """Create from a vector."""
        feat_counts = [len(type_features[o.type]) for o in constant_objects][:-1]
        splits = np.cumsum(feat_counts)
        data = dict(zip(constant_objects, np.split(vec, splits)))
        return cls(data, type_features)

    def copy(self) -> Self:
        """Return a copy of this state."""
        new_data = {}
        for obj in self:
            new_data[obj] = self._copy_state_value(self.data[obj])
        return type(self)(new_data, self.type_features)

    def _copy_state_value(self, val: Any) -> Any:
        if val is None or isinstance(val, (float, bool, int, str)):
            return val
        if isinstance(val, (list, tuple, set)):
            return type(val)(self._copy_state_value(v) for v in val)
        assert hasattr(val, "copy")
        return val.copy()

    def allclose(self, other: Self, atol: float = 1e-3) -> bool:
        """Return whether this state is close enough to another one, i.e., its
        objects are the same, and the features are close."""
        if not sorted(self.data) == sorted(other.data):
            return False
        for obj in self.data:
            if not np.allclose(self.data[obj], other.data[obj], atol=atol):
                return False
        return True

    def pretty_str(self) -> str:
        """Display the state in a nice human-readable format."""
        type_to_table: Dict[Type, List[List[str]]] = {}
        for obj in self:
            if obj.type not in type_to_table:
                type_to_table[obj.type] = []
            type_to_table[obj.type].append([obj.name] + list(map(str, self[obj])))
        table_strs = []
        for t in sorted(type_to_table):
            headers = ["type: " + t.name] + list(self.type_features[t])
            table_strs.append(tabulate(type_to_table[t], headers=headers))
        ll = max(len(line) for table in table_strs for line in table.split("\n"))
        prefix = "#" * (ll // 2 - 3) + " STATE " + "#" * (ll - ll // 2 - 4) + "\n"
        suffix = "\n" + "#" * ll + "\n"
        return prefix + "\n\n".join(table_strs) + suffix


# Constants.
DefaultObjectCentricState = ObjectCentricState({}, {})
