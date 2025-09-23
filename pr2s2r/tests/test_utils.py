"""Tests for utils.py."""

from pr2s2r.structs import Dog
from pr2s2r.utils import get_good_dogs_of_breed


def test_get_good_dogs_of_breed():
    """Tests for get_good_dogs_of_breed()."""
    ralph = Dog("ralph", "corgi")
    maeve = Dog("maeve", "corgi")
    frank = Dog("frank", "poodle")
    dogs = {ralph, maeve, frank}
    assert get_good_dogs_of_breed(dogs, "corgi") == {ralph, maeve}
