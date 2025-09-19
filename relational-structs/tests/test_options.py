"""Tests for options.py."""

import numpy as np
import pytest

from relational_structs import (
    DefaultObjectCentricState,
    ObjectCentricState,
    ObjectSequenceBoxSpace,
    Option,
    ParameterizedOption,
    Type,
)


def test_option():
    """Tests for ParameterizedOption, Option classes."""

    type1 = Type("type1")
    type2 = Type("type2")
    type_to_feats = {type1: ["feat1", "feat2"], type2: ["feat3", "feat4", "feat5"]}

    params_space = ObjectSequenceBoxSpace([type1, type2], low=[-10, -10], high=[10, 10])

    def policy(s, p, m):
        del s, m  # unused
        return p[1] * 2

    def initiable(s, p, m):
        m["test_key"] = "test_string"
        obj = list(s)[0]
        return p[1][0] < s[obj][0]

    def terminal(s, p, m):
        del m  # unused
        obj = list(s)[0]
        return p[1][1] > s[obj][2]

    parameterized_option = ParameterizedOption(
        "Pick", params_space, policy, initiable, terminal
    )

    assert (
        repr(parameterized_option)
        == str(parameterized_option)
        == "ParameterizedOption(name='Pick')"
    )

    obj1 = type1("obj1")
    obj2 = type2("obj2")
    obj_params = [obj1, obj2]
    box_params = np.array([-15, 5], dtype=np.float32)
    params = (obj_params, box_params)
    with pytest.raises(AssertionError):
        parameterized_option.ground(params)  # box params not in params_space
    box_params = np.array([-5, 5], dtype=np.float32)
    params = (obj_params, box_params)
    option = parameterized_option.ground(params)
    assert isinstance(option, Option)
    assert (
        repr(option)
        == str(option)
        == "Option(name='Pick', params=([obj1:type1, obj2:type2], array([-5.,  5.], dtype=float32)))"  # pylint: disable=line-too-long
    )
    assert option.name == "Pick"
    assert option.memory == {}
    assert option.parent.name == "Pick"
    assert option.parent is parameterized_option

    obj3 = type1("obj3")
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    obj2 = type2("obj2")
    obj4 = type2("obj4")
    obj9 = type2("obj9")
    state = ObjectCentricState(
        {
            obj3: [1, 2],
            obj7: [3, 4],
            obj1: [5, 6, 7],
            obj4: [8, 9, 10],
            obj9: [11, 12, 13],
        },
        type_to_feats,
    )

    assert np.all(option.policy(state) == np.array(box_params) * 2)
    assert option.initiable(state)
    assert option.memory == {"test_key": "test_string"}  # set by initiable()
    assert not option.terminal(state)
    assert option.params[1][0] == -5 and option.params[1][1] == 5
    box_params = [5, -5]
    params = (obj_params, box_params)
    option = parameterized_option.ground(params)
    assert isinstance(option, Option)
    assert option.params[1][0] == 5 and option.params[1][1] == -5

    parameterized_option2 = ParameterizedOption(
        "Pick2", params_space, policy, initiable, terminal
    )
    assert parameterized_option2 > parameterized_option
    assert parameterized_option < parameterized_option2


def test_option_memory_incorrect():
    """Tests for doing option memory the WRONG way.

    Ensures that it fails in the way we'd expect.
    """

    def _make_option():
        value = 0.0

        def policy(s, p, m):
            del s  # unused
            del m  # the correct way of doing memory is unused here
            nonlocal value
            value += p[1][0]  # add the param to value
            return p[1]

        return ParameterizedOption(
            "Dummy",
            ObjectSequenceBoxSpace([], [0], [1]),
            policy,
            lambda s, p, m: True,
            lambda s, p, m: value > 1.0,
        )  # terminate when value > 1.0

    param_opt = _make_option()
    opt1 = param_opt.ground(([], [0.7]))
    opt2 = param_opt.ground(([], [0.4]))
    state = DefaultObjectCentricState
    assert abs(opt1.policy(state)[0] - 0.7) < 1e-6
    assert abs(opt2.policy(state)[0] - 0.4) < 1e-6
    # Since memory is shared between the two ground options, both will be
    # terminal now, since they'll share a value of 1.1 -- this is BAD, but
    # we include this test as an example of what NOT to do.
    assert opt1.terminal(state)
    assert opt2.terminal(state)


def test_option_memory_correct():
    """Tests for doing option memory the RIGHT way.

    Uses the memory dict.
    """

    def _make_option():

        def initiable(s, p, m):
            del s, p  # unused
            m["value"] = 0.0  # initialize value
            return True

        def policy(s, p, m):
            del s  # unused
            assert "value" in m, "Call initiable() first!"
            m["value"] += p[1][0]  # add the param to value
            return p[1]

        return ParameterizedOption(
            "Dummy",
            ObjectSequenceBoxSpace([], [0], [1]),
            policy,
            initiable,
            lambda s, p, m: m["value"] > 1.0,
        )  # terminate when value > 1.0

    param_opt = _make_option()
    opt1 = param_opt.ground(([], [0.7]))
    opt2 = param_opt.ground(([], [0.4]))
    state = DefaultObjectCentricState
    assert opt1.initiable(state)
    assert opt2.initiable(state)
    assert abs(opt1.policy(state)[0] - 0.7) < 1e-6
    assert abs(opt2.policy(state)[0] - 0.4) < 1e-6
    # Since memory is NOT shared between the two ground options, neither
    # will be terminal now.
    assert not opt1.terminal(state)
    assert not opt2.terminal(state)
    # Now make opt1 terminal.
    assert abs(opt1.policy(state)[0] - 0.7) < 1e-6
    assert opt1.terminal(state)
    assert not opt2.terminal(state)
    # opt2 is not quite terminal yet...value is 0.8
    opt2.policy(state)
    assert not opt2.terminal(state)
    # Make opt2 terminal.
    opt2.policy(state)
    assert opt2.terminal(state)
