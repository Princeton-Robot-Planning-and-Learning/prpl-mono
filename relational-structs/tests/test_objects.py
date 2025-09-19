"""Tests for objects.py."""

import pytest

from relational_structs import Object, Type, Variable


def test_object_type():
    """Tests for Type class."""
    name = "test"
    my_type = Type(name)
    assert my_type.name == name
    assert isinstance(hash(my_type), int)
    name = "test2"
    my_type2 = Type(name, parent=my_type)
    assert my_type2.name == name
    assert isinstance(hash(my_type2), int)
    assert my_type2.parent == my_type
    name = "test2"
    my_type3 = Type(name, parent=my_type)  # same as my_type2
    obj = my_type("obj1")
    assert obj.is_instance(my_type)
    assert not obj.is_instance(my_type2)
    assert not obj.is_instance(my_type3)
    obj = my_type2("obj2")
    assert obj.is_instance(my_type)
    assert obj.is_instance(my_type2)
    assert obj.is_instance(my_type3)


def test_object():
    """Tests for Object class."""
    my_name = "obj"
    my_type = Type("type")
    obj = my_type(my_name)
    assert isinstance(obj, Object)
    assert obj.name == my_name
    assert obj.type == my_type
    assert str(obj) == repr(obj) == "obj:type"
    assert isinstance(hash(obj), int)
    with pytest.raises(AssertionError):
        Object("?obj", my_type)  # name cannot start with ?


def test_variable():
    """Tests for Variable class."""
    my_name = "?var"
    my_type = Type("type", ["feat1", "feat2"])
    var = my_type(my_name)
    assert isinstance(var, Variable)
    assert var.name == my_name
    assert var.type == my_type
    assert str(var) == repr(var) == "?var:type"
    assert isinstance(hash(var), int)
    with pytest.raises(AssertionError):
        Variable("var", my_type)  # name must start with ?
