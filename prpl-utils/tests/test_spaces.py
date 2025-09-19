"""Tests for spaces.py."""

from prpl_utils.spaces import EnumSpace, FunctionalSpace


def test_enum_space():
    """Tests for EnumSpace()."""
    elements = ["cat", "dog", "horse"]
    space = EnumSpace(elements)
    space.seed(123)
    elm = space.sample()
    assert elm in elements
    assert space.contains(elm)
    assert not space.is_np_flattenable


def test_functional_space():
    """Tests for FunctionalSpace()."""

    def _sample_fn(rng):
        return float(rng.uniform(-3, 3))

    def _contains_fn(x):
        if not isinstance(x, float):
            return False
        return -3 <= x <= 3

    space = FunctionalSpace(_contains_fn, _sample_fn, seed=123)
    for _ in range(10):
        x = space.sample()
        assert space.contains(x)
