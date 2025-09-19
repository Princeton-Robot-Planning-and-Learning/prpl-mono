"""Tests for state.py."""

import numpy as np
import pytest

from relational_structs import (
    ObjectCentricState,
    Type,
)


def test_state():
    """Tests for State class."""
    type1 = Type("type1")
    type2 = Type("type2")
    type_to_features = {type1: ["feat1", "feat2"], type2: ["feat3", "feat4", "feat5"]}
    obj3 = type1("obj3")
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    obj2 = type2("obj2")
    obj1_dup = type2("obj1")
    obj4 = type2("obj4")
    obj9 = type2("obj9")
    assert obj7 > obj1
    assert obj1 < obj4
    assert obj1 < obj3
    assert obj1 != obj9
    assert obj1 == obj1_dup
    with pytest.raises(AssertionError):
        ObjectCentricState(
            {obj3: [1, 2, 3]}, type_to_features
        )  # bad feature vector dimension
    state = ObjectCentricState(
        {
            obj3: [1, 2],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        },
        type_to_features,
    )
    sorted_objs = list(state)
    assert sorted_objs == [obj1, obj3, obj4, obj7, obj9]
    assert state[obj9] == state.data[obj9] == [11, 12, 13]
    assert state.get(obj3, "feat2") == 2
    assert state.get(obj1, "feat4") == 6
    with pytest.raises(ValueError):
        state.get(obj3, "feat3")  # feature not in list
    with pytest.raises(ValueError):
        state.get(obj1, "feat1")  # feature not in list
    assert state.get_object_from_name("obj1") == obj1
    with pytest.raises(ValueError):
        assert state.get_object_from_name("not a real object")
    vec = state.vec([obj3, obj1])
    assert vec.shape == (5,)
    assert list(vec) == [1, 2, 5, 6, 7]
    state.set(obj3, "feat2", 122)
    assert state.get(obj3, "feat2") == 122
    state2 = state.copy()
    assert state.allclose(state2)
    state2[obj1][0] = 999
    state2.set(obj1, "feat5", 991)
    assert state != state2  # changing copy doesn't change original
    assert state2.get(obj1, "feat3") == 999
    assert state2[obj1][2] == 991
    state3 = ObjectCentricState({obj3: np.array([1, 2])}, type_to_features)
    state3.copy()  # try copying with numpy array
    # Test state vec with no objects
    vec = state.vec([])
    assert vec.shape == (0,)
    # Test allclose
    state2 = ObjectCentricState(
        {
            obj3: [1, 122],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        },
        type_to_features,
    )
    assert state.allclose(state2)
    state2 = ObjectCentricState(
        {
            obj3: [1, 122],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8.3, 9, 10],
            obj9: [11, 12, 13],
        },
        type_to_features,
    )
    assert not state.allclose(state2)  # obj4 state is different
    state2 = ObjectCentricState(
        {obj3: [1, 122], obj7: [3, 4], obj4: [8, 9, 10], obj9: [11, 12, 13]},
        type_to_features,
    )
    assert not state.allclose(state2)  # obj1 is missing
    state2 = ObjectCentricState(
        {
            obj3: [1, 122],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj2: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        },
        type_to_features,
    )
    assert not state.allclose(state2)  # obj2 is extra
    # Test pretty_str
    assert (
        state2.pretty_str()
        == """################# STATE ################
type: type1      feat1    feat2
-------------  -------  -------
obj3                 1      122
obj7                 3        4

type: type2      feat3    feat4    feat5
-------------  -------  -------  -------
obj1                 5        6        7
obj2                 5        6        7
obj4                 8        9       10
obj9                11       12       13
########################################
"""
    )
    # Test hashing.
    assert isinstance(hash(state), int)
    assert hash(state) != hash(state2)


def test_state_inheritance():
    """Tests for subclasses of ObjectCentricState."""

    class _CustomObjectCentricState(ObjectCentricState):
        """Custom class."""

        def hello(self) -> str:
            """For testing."""
            return "world"

    type1 = Type("type1")
    type_to_features = {type1: ["feat1", "feat2"]}
    obj3 = type1("obj3")
    obj7 = type1("obj7")

    state = _CustomObjectCentricState(
        {
            obj3: [1, 2],
            obj7: [3, 4],
        },
        type_to_features,
    )

    state_copy = state.copy()
    assert isinstance(state_copy, _CustomObjectCentricState)
    assert state_copy.allclose(state)
    assert state_copy.hello() == "world"
