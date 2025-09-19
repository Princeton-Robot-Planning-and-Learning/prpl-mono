"""Tests for spaces.py."""

import pytest

from relational_structs import ObjectCentricState, ObjectCentricStateSpace, Type, utils


def test_object_centric_state_space():
    """Tests for ObjectCentricStateSpace."""
    my_type = Type("test")
    my_type2 = Type("test2")
    type_to_feats = {
        my_type: ["feat1", "feat2"],
        my_type2: ["feat3"],
    }
    types = {my_type, my_type2}
    space = ObjectCentricStateSpace(types)
    assert not space.is_np_flattenable
    assert space.types == types
    obj1 = my_type("obj1")
    obj2 = my_type("obj2")
    obj3 = my_type2("obj3")
    state = utils.create_state_from_dict(
        {
            obj1: {
                "feat1": 1,
                "feat2": 1,
            },
            obj2: {
                "feat1": 1,
                "feat2": 0,
            },
            obj3: {
                "feat3": 1,
            },
        },
        type_to_feats,
    )
    assert space.contains(state)
    my_type3 = Type("test3")
    type_to_feats[my_type3] = ["feat4"]
    obj4 = my_type3("obj4")
    state2 = utils.create_state_from_dict(
        {
            obj1: {
                "feat1": 1,
                "feat2": 1,
            },
            obj4: {
                "feat4": 1,
            },
        },
        type_to_feats,
    )
    assert not space.contains(state2)
    with pytest.raises(NotImplementedError):
        space.sample()

    # Test conversion to ObjectCentricBoxSpace().
    box = space.to_box([obj1, obj2, obj3], type_to_feats)
    assert box.shape == (5,)
    vec = box.vectorize(state)
    assert vec.shape == (5,)
    assert box.contains(vec)
    recovered_state = box.devectorize(vec)
    assert recovered_state.allclose(state)

    # Test with subclasses of ObjectCentricState.
    class _CustomObjectCentricState(ObjectCentricState):
        """Custom class."""

        def hello(self) -> str:
            """For testing."""
            return "world"

    custom_space = ObjectCentricStateSpace(types, state_cls=_CustomObjectCentricState)
    box = custom_space.to_box([obj1, obj2, obj3], type_to_feats)
    assert box.shape == (5,)
    vec = box.vectorize(state)
    assert vec.shape == (5,)
    assert box.contains(vec)
    recovered_state = box.devectorize(vec)
    assert recovered_state.allclose(state)
    assert isinstance(recovered_state, _CustomObjectCentricState)
