"""Common utility classes and functions."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Collection, Iterator

from pyperplan.heuristics.heuristic_base import Heuristic as _PyperplanBaseHeuristic
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    PDDLDomain,
    PDDLProblem,
    Predicate,
)
from relational_structs.utils import all_ground_operators, get_object_combinations

from bilevel_planning.structs import (
    LiftedSkill,
    ParameterizedController,
    RelationalAbstractState,
)

############################### Pyperplan Glue ###############################
# NOTE: this code should probably be relocated, but I don't know to where yet.


def get_static_atoms(
    ground_ops: Collection[GroundOperator], atoms: Collection[GroundAtom]
) -> set[GroundAtom]:
    """Get the subset of atoms from the given set that are static with respect to the
    given ground operators."""
    static_atoms = set(atoms)
    for ground_op in ground_ops:
        for atom in ground_op.add_effects | ground_op.delete_effects:
            static_atoms.discard(atom)
    return static_atoms


def get_all_ground_atoms_for_predicate(
    predicate: Predicate, objects: Collection[Object]
) -> set[GroundAtom]:
    """Get all groundings of the predicate given objects."""
    ground_atoms = set()
    for args in get_object_combinations(objects, predicate.types):
        ground_atom = GroundAtom(predicate, args)
        ground_atoms.add(ground_atom)
    return ground_atoms


def create_pyperplan_heuristic(
    heuristic_name: str,
    pddl_domain: PDDLDomain,
    pddl_problem: PDDLProblem,
    ground_ops: set[GroundOperator],
) -> PyperplanHeuristicWrapper:
    """Create a pyperplan heuristic."""
    assert heuristic_name in _PYPERPLAN_HEURISTICS
    static_atoms = get_static_atoms(ground_ops, pddl_problem.init_atoms)
    pyperplan_heuristic_cls = _PYPERPLAN_HEURISTICS[heuristic_name]
    pyperplan_task = _create_pyperplan_task(
        set(pddl_problem.init_atoms),
        set(pddl_problem.goal),
        ground_ops,
        pddl_domain.predicates,
        pddl_problem.objects,
        static_atoms,
    )
    pyperplan_heuristic = pyperplan_heuristic_cls(pyperplan_task)
    pyperplan_goal = _atoms_to_pyperplan_facts(set(pddl_problem.goal) - static_atoms)
    return PyperplanHeuristicWrapper(
        heuristic_name,
        pddl_problem.init_atoms,
        set(pddl_problem.goal),
        ground_ops,
        static_atoms,
        pyperplan_heuristic,
        pyperplan_goal,
    )


_PyperplanFacts = frozenset[str]


@dataclass(frozen=True)
class _PyperplanNode:
    """Container glue for pyperplan heuristics."""

    state: _PyperplanFacts
    goal: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanOperator:
    """Container glue for pyperplan heuristics."""

    name: str
    preconditions: _PyperplanFacts
    add_effects: _PyperplanFacts
    del_effects: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanTask:
    """Container glue for pyperplan heuristics."""

    facts: _PyperplanFacts
    initial_state: _PyperplanFacts
    goals: _PyperplanFacts
    operators: Collection[_PyperplanOperator]


@dataclass(frozen=True)
class PyperplanHeuristicWrapper:
    """A light wrapper around pyperplan's heuristics."""

    name: str
    init_atoms: Collection[GroundAtom]
    goal: set[GroundAtom]
    ground_ops: Collection[GroundOperator]
    _static_atoms: set[GroundAtom]
    _pyperplan_heuristic: _PyperplanBaseHeuristic
    _pyperplan_goal: _PyperplanFacts

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        # Note: filtering out static atoms.
        pyperplan_facts = _atoms_to_pyperplan_facts(set(atoms) - self._static_atoms)
        return self._evaluate(
            pyperplan_facts, self._pyperplan_goal, self._pyperplan_heuristic
        )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _evaluate(
        pyperplan_facts: _PyperplanFacts,
        pyperplan_goal: _PyperplanFacts,
        pyperplan_heuristic: _PyperplanBaseHeuristic,
    ) -> float:
        pyperplan_node = _PyperplanNode(pyperplan_facts, pyperplan_goal)
        logging.disable(logging.DEBUG)
        result = pyperplan_heuristic(pyperplan_node)
        logging.disable(logging.NOTSET)
        return result


def _create_pyperplan_task(
    init_atoms: set[GroundAtom],
    goal: set[GroundAtom],
    ground_ops: Collection[GroundOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
    static_atoms: set[GroundAtom],
) -> _PyperplanTask:
    """Helper glue for pyperplan heuristics."""
    all_atoms = set()
    for predicate in predicates:
        all_atoms.update(
            get_all_ground_atoms_for_predicate(predicate, frozenset(objects))
        )
    # Note: removing static atoms.
    pyperplan_facts = _atoms_to_pyperplan_facts(all_atoms - static_atoms)
    pyperplan_state = _atoms_to_pyperplan_facts(init_atoms - static_atoms)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    pyperplan_operators = set()
    for op in ground_ops:
        # Note: the pyperplan operator must include the objects, because hFF
        # uses the operator name in constructing the relaxed plan, and the
        # relaxed plan is a set. If we instead just used op.name, there would
        # be a very nasty bug where two ground operators in the relaxed plan
        # that have different objects are counted as just one.
        name = op.name + "-".join(o.name for o in op.parameters)
        pyperplan_operator = _PyperplanOperator(
            name,
            # Note: removing static atoms from preconditions.
            _atoms_to_pyperplan_facts(op.preconditions - static_atoms),
            _atoms_to_pyperplan_facts(op.add_effects),
            _atoms_to_pyperplan_facts(op.delete_effects),
        )
        pyperplan_operators.add(pyperplan_operator)
    return _PyperplanTask(
        pyperplan_facts, pyperplan_state, pyperplan_goal, pyperplan_operators
    )


@functools.lru_cache(maxsize=None)
def _atom_to_pyperplan_fact(atom: GroundAtom) -> str:
    """Convert atom to tuple for interface with pyperplan."""
    arg_str = " ".join(o.name for o in atom.objects)
    return f"({atom.predicate.name} {arg_str})"


def _atoms_to_pyperplan_facts(atoms: Collection[GroundAtom]) -> _PyperplanFacts:
    """Light wrapper around _atom_to_pyperplan_fact() that operates on a collection of
    atoms."""
    return frozenset({_atom_to_pyperplan_fact(atom) for atom in atoms})


############################## End Pyperplan Glue ##############################


def cached_all_ground_operators(
    operators: Collection[LiftedOperator], objects: Collection[Object]
) -> set[GroundOperator]:
    """Get all ground operators, with caching."""
    return _cached_all_ground_operators(frozenset(operators), frozenset(objects))


@functools.lru_cache(maxsize=None)
def _cached_all_ground_operators(
    operators: frozenset[LiftedOperator], objects: frozenset[Object]
) -> set[GroundOperator]:
    return all_ground_operators(operators, objects)


@dataclass(frozen=True)
class RelationalAbstractSuccessorGenerator:
    """A successor generator that uses relational states and actions."""

    operators: set[LiftedOperator]

    def __call__(
        self, abstract_state: RelationalAbstractState
    ) -> Iterator[tuple[GroundOperator, RelationalAbstractState]]:
        ground_operators = cached_all_ground_operators(
            self.operators, abstract_state.objects
        )
        for ground_operator in ground_operators:
            if ground_operator.preconditions.issubset(abstract_state.atoms):
                next_atoms = (
                    abstract_state.atoms - ground_operator.delete_effects
                ) | ground_operator.add_effects
                next_abstract_state = RelationalAbstractState(
                    next_atoms, abstract_state.objects
                )
                yield (ground_operator, next_abstract_state)


class RelationalControllerGenerator:
    """Generates controllers by looking up the corresponding skill."""

    def __init__(self, skills: set[LiftedSkill]) -> None:
        self._operator_to_controller = {s.operator: s.controller for s in skills}

    def __call__(self, abstract_action: GroundOperator) -> ParameterizedController:
        assert abstract_action.parent is not None
        lifted_controller = self._operator_to_controller[abstract_action.parent]
        return lifted_controller.ground(abstract_action.parameters)
