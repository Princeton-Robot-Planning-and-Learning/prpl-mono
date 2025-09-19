"""Utilities."""

import itertools
from typing import (
    Collection,
    Dict,
    Iterator,
    List,
    Sequence,
    Set,
)
from typing import Type as TypingType
from typing import (
    TypeVar,
)

import numpy as np

from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object, Type, TypedEntity
from relational_structs.pddl import (
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
)


def create_state_from_dict(
    data: Dict[Object, Dict[str, float]],
    type_features: Dict[Type, List[str]],
    state_cls: TypingType[ObjectCentricState] = ObjectCentricState,
) -> ObjectCentricState:
    """Small utility to generate a state from a dictionary `data` of individual
    feature values for each object."""
    state_dict = {}
    for obj, obj_data in data.items():
        obj_vec = []
        for feat in type_features[obj.type]:
            obj_vec.append(obj_data[feat])
        state_dict[obj] = np.array(obj_vec)
    return state_cls(state_dict, type_features)


_TypedEntityTypeVar = TypeVar("_TypedEntityTypeVar", bound=TypedEntity)


def _get_entity_combinations(
    entities: Collection[_TypedEntityTypeVar], types: Sequence[Type]
) -> Iterator[List[_TypedEntityTypeVar]]:
    """Get all combinations of entities satisfying the given types sequence."""
    sorted_entities = sorted(entities)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_object_combinations(
    objects: Collection[Object], types: Sequence[Type]
) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(objects, types)


def parse_pddl_plan(
    ground_op_strs: List[str], domain: PDDLDomain, problem: PDDLProblem
) -> List[GroundOperator]:
    """Parse a plan in string form into ground operator form."""
    ground_op_plan: List[GroundOperator] = []
    op_name_to_op = {o.name.lower(): o for o in domain.operators}
    obj_name_to_obj = {obj.name.lower(): obj for obj in problem.objects}
    for s in ground_op_strs:
        assert s[0] == "("
        assert s[-1] == ")"
        s = s[1:-1]
        op_name, s = s.split(" ", maxsplit=1)
        op_name = op_name.lower()
        assert op_name in op_name_to_op, f"Unknown operator name {op_name}"
        op = op_name_to_op[op_name]
        objs: List[Object] = []
        for obj_name in s.split(" "):
            obj_name = obj_name.lower()
            assert obj_name in obj_name_to_obj, f"Unknown object name {obj_name}"
            objs.append(obj_name_to_obj[obj_name])
        ground_op = op.ground(tuple(objs))
        ground_op_plan.append(ground_op)
    return ground_op_plan


def all_ground_operators(
    operators: Collection[LiftedOperator], objects: Collection[Object]
) -> Set[GroundOperator]:
    """Get all ground operators given objects."""
    ground_operators: Set[GroundOperator] = set()
    for operator in operators:
        types = [p.type for p in operator.parameters]
        for choice in get_object_combinations(objects, types):
            ground_operator = operator.ground(tuple(choice))
            ground_operators.add(ground_operator)
    return ground_operators
