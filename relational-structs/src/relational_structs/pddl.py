"""PDDL related data structures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from graphlib import TopologicalSorter
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from multimethod import multimethod
from prpl_utils.utils import consistent_hash
from pyperplan.pddl.parser import (
    TraversePDDLDomain,
    TraversePDDLProblem,
    parse_domain_def,
    parse_lisp_iterator,
    parse_problem_def,
)
from pyperplan.pddl.pddl import Domain as PyperplanDomain
from pyperplan.pddl.pddl import Type as PyperplanType

from relational_structs.objects import Object, Type, TypedEntity, Variable

_TypedEntityTypeVar = TypeVar("_TypedEntityTypeVar", bound=TypedEntity)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom(Generic[_TypedEntityTypeVar]):
    """Struct defining an atom (a predicate applied to either variables or
    objects).

    Should not be instantiated externally.
    """

    predicate: Predicate
    entities: Sequence[_TypedEntityTypeVar]

    def __post_init__(self) -> None:
        if isinstance(self.entities, TypedEntity):
            raise ValueError(
                "Atoms expect a sequence of entities, not a single entity."
            )
        assert len(self.entities) == self.predicate.arity
        for ent, pred_type in zip(self.entities, self.predicate.types):
            assert ent.is_instance(pred_type)

    @cached_property
    def _hash(self) -> int:
        return consistent_hash(str(self))

    @cached_property
    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        if not self.entities:
            return f"({self.predicate.name})"
        entities_str = " ".join(e.name for e in self.entities)
        return f"({self.predicate.name} {entities_str})"

    def __str__(self) -> str:
        return self.pddl_str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, _Atom)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) < str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom[Variable]):
    """Struct defining a lifted atom (a predicate applied to variables)."""

    @cached_property
    def variables(self) -> List[Variable]:
        """Arguments for this lifted atom."""
        return list(self.entities)

    def ground(self, sub: Dict[Variable, Object]) -> GroundAtom:
        """Create a GroundAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return GroundAtom(self.predicate, [sub[v] for v in self.variables])

    def substitute(self, sub: Dict[Variable, Variable]) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[v] for v in self.variables])


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom[Object]):
    """Struct defining a ground atom (a predicate applied to objects)."""

    @cached_property
    def objects(self) -> List[Object]:
        """Arguments for this ground atom."""
        return list(self.entities)

    def lift(self, sub: Dict[Object, Variable]) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.objects).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[o] for o in self.objects])


@dataclass(frozen=True, order=True, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states)."""

    name: str
    types: Sequence[Type]

    @multimethod
    def __call__(self, entities: Sequence[TypedEntity]) -> Any:
        raise NotImplementedError(f"Cannot call predicate with {entities}")

    @__call__.register
    def _(self, entities: Sequence[Variable]) -> LiftedAtom:
        return LiftedAtom(self, entities)

    @__call__.register
    def _(  # pylint: disable=function-redefined
        self, entities: Sequence[Object]
    ) -> GroundAtom:
        return GroundAtom(self, entities)

    @cached_property
    def _hash(self) -> int:
        return consistent_hash(str(self))

    @cached_property
    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to PDDL."""
        if self.arity == 0:
            return f"({self.name})"
        vars_str = " ".join(f"?x{i} - {t.name}" for i, t in enumerate(self.types))
        return f"({self.name} {vars_str})"

    @cached_property
    def arity(self) -> int:
        """The arity of this predicate (number of arguments)."""
        return len(self.types)

    def __str__(self) -> str:
        return self.pddl_str

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return str(self)


_AtomTypeVar = TypeVar("_AtomTypeVar", bound=_Atom)


@dataclass(frozen=True, repr=False, eq=False)
class Operator(Generic[_TypedEntityTypeVar, _AtomTypeVar]):
    """Struct defining a symbolic operator (as in STRIPS)."""

    name: str
    parameters: Sequence[_TypedEntityTypeVar]
    preconditions: Set[_AtomTypeVar]
    add_effects: Set[_AtomTypeVar]
    delete_effects: Set[_AtomTypeVar]
    cost: Optional[int] = None

    def __post_init__(self) -> None:
        # Verify that all of the entities in the preconditions and effects are
        # listed as parameters.
        for atom in self.preconditions | self.add_effects | self.delete_effects:
            for entity in atom.entities:
                assert (
                    entity in self.parameters
                ), f"{entity} is missing from operator parameters"

    @cached_property
    def _hash(self) -> int:
        return consistent_hash(str(self))

    @cached_property
    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to PDDL."""
        params_str = " ".join(f"{p.name} - {p.type.name}" for p in self.parameters)
        preconds_str = "\n        ".join(
            atom.pddl_str for atom in sorted(self.preconditions)
        )
        effects_str = "\n        ".join(
            atom.pddl_str for atom in sorted(self.add_effects)
        )
        if self.delete_effects:
            effects_str += "\n        "
            effects_str += "\n        ".join(
                f"(not {atom.pddl_str})" for atom in sorted(self.delete_effects)
            )
        # Add cost effect if cost is specified
        if self.cost is not None:
            if effects_str:
                effects_str += "\n        "
            effects_str += f"(increase (total-cost) {self.cost})"
        return f"""(:action {self.name}
    :parameters ({params_str})
    :precondition (and {preconds_str})
    :effect (and {effects_str})
)"""

    @cached_property
    def short_str(self) -> str:
        """Abbreviated name, not necessarily unique."""
        param_str = ", ".join([p.name for p in self.parameters])
        return f"{self.name}({param_str})"

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return self.pddl_str

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, Operator)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Operator)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, Operator)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedOperator(Operator[Variable, LiftedAtom]):
    """Struct defining a lifted symbolic operator (as in STRIPS)."""

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> GroundOperator:
        """Ground into a _GroundSTRIPSOperator, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        return GroundOperator(
            self.name,
            list(objects),
            preconditions,
            add_effects,
            delete_effects,
            self.cost,
            self,
        )


@dataclass(frozen=True, repr=False, eq=False)
class GroundOperator(Operator[Object, GroundAtom]):
    """A STRIPSOperator + objects."""

    parent: Optional[LiftedOperator] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.parent is None:
            raise ValueError("GroundOperator must have a parent LiftedOperator")


@lru_cache(maxsize=None)
def _domain_str_to_pyperplan_domain(domain_str: str) -> PyperplanDomain:
    domain_ast = parse_domain_def(parse_lisp_iterator(domain_str.split("\n")))
    visitor = TraversePDDLDomain()
    domain_ast.accept(visitor)
    return visitor.domain


def _extract_action_costs_from_pddl(pddl_str: str) -> Dict[str, int]:
    """Extract action costs from PDDL string.

    Looks for patterns like:
    (increase (total-cost) 5)
    """
    action_costs = {}
    action_sections = re.split(r":action\s+(\w+)", pddl_str, flags=re.IGNORECASE)

    # Skip the first element (before first action) and process pairs
    for i in range(1, len(action_sections), 2):
        if i + 1 < len(action_sections):
            action_name = action_sections[i]
            action_content = action_sections[i + 1]
            # Look for cost increase patterns in the entire action content
            cost_pattern = r"\(increase\s*\(\s*total-cost\s*\)\s*([0-9]+)\s*\)"
            cost_match = re.search(cost_pattern, action_content, re.IGNORECASE)
            if cost_match:
                try:
                    cost = int(cost_match.group(1))
                    action_costs[action_name.lower()] = cost
                except ValueError:
                    # If we can't parse the cost as an int, skip it
                    pass
    return action_costs


def _remove_action_costs_from_domain_str(domain_str: str) -> str:
    """Remove action cost related elements from a domain string for pyperplan
    parsing."""
    # Remove action-costs requirement
    domain_str = re.sub(r":action-costs\s*", "", domain_str, flags=re.IGNORECASE)
    # Remove functions section
    lines = domain_str.split("\n")
    filtered_lines = []
    in_functions = False
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("(:functions"):
            in_functions = True
            paren_count = 1  # Start counting from the opening paren
            for char in line[line.find("(:functions") + len("(:functions") :]:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        in_functions = False
                        break
            continue
        if in_functions:
            for char in line:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        in_functions = False
                        break
            continue
        filtered_lines.append(line)

    domain_str = "\n".join(filtered_lines)
    # Remove cost increase effects
    domain_str = re.sub(
        r"\(increase\s*\(\s*total-cost\s*\)\s*[0-9]+\s*\)",
        "",
        domain_str,
        flags=re.IGNORECASE,
    )
    # Clean up any resulting empty lines or extra whitespace
    domain_str = re.sub(r"\n\s*\n", "\n", domain_str)
    return domain_str


def _check_uses_action_costs(pddl_str: str) -> bool:
    """Check if PDDL domain uses action costs."""
    # Check for action-costs requirement
    if re.search(r":requirements.*:action-costs", pddl_str, re.IGNORECASE | re.DOTALL):
        return True
    # Check for total-cost function
    if re.search(r"total-cost", pddl_str, re.IGNORECASE):
        return True
    # Check for cost increase in effects
    if re.search(r"\(increase\s*\(\s*total-cost\s*\)", pddl_str, re.IGNORECASE):
        return True
    return False


def _remove_action_costs_from_problem_str(problem_str: str) -> str:
    """Remove total-cost initializations and metric sections from a problem
    string so pyperplan can parse it."""
    # Remove initial total-cost assignments like (= (total-cost) 0)
    problem_str = re.sub(
        r"\(\s*=\s*\(\s*total-cost\s*\)\s*[0-9]+\s*\)",
        "",
        problem_str,
        flags=re.IGNORECASE,
    )
    # Remove metric section like (:metric minimize (total-cost)) possibly
    # spanning multiple lines.
    problem_str = re.sub(
        r"\(:metric\b.*?total-cost.*?\)",
        "",
        problem_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Clean up any resulting empty lines or extra whitespace
    problem_str = re.sub(r"\n\s*\n", "\n", problem_str)
    # If removals left unmatched closing parentheses, trim trailing ')' until
    # parentheses are balanced. This avoids creating malformed Lisp for
    # downstream parsers.
    open_paren = problem_str.count("(")
    close_paren = problem_str.count(")")
    while close_paren > open_paren:
        # strip trailing whitespace then a single trailing ')'
        problem_str = problem_str.rstrip()
        if problem_str.endswith(")"):
            problem_str = problem_str[:-1]
        else:
            break
        open_paren = problem_str.count("(")
        close_paren = problem_str.count(")")
    # Final cleanup of extra blank lines introduced by trimming.
    problem_str = re.sub(r"\n\s*\n", "\n", problem_str)
    return problem_str


@dataclass(frozen=True)
class PDDLDomain:
    """A PDDL domain."""

    name: str
    operators: Collection[LiftedOperator]
    predicates: Collection[Predicate]
    types: Collection[Type]
    uses_action_costs: bool = False

    @classmethod
    def parse(cls, pddl_str: str) -> PDDLDomain:
        """Parse a domain from a string."""
        uses_action_costs = _check_uses_action_costs(pddl_str)
        action_costs = (
            _extract_action_costs_from_pddl(pddl_str) if uses_action_costs else {}
        )
        # Create a version of the PDDL without cost-related elements for pyperplan
        pddl_for_parsing = pddl_str
        if uses_action_costs:
            pddl_for_parsing = _remove_action_costs_from_domain_str(pddl_str)

        # Let pyperplan do most of the heavy lifting.
        pyperplan_domain = _domain_str_to_pyperplan_domain(pddl_for_parsing)
        domain_name = pyperplan_domain.name
        pyperplan_types = pyperplan_domain.types
        pyperplan_predicates = pyperplan_domain.predicates
        pyperplan_operators = pyperplan_domain.actions
        # Convert the pyperplan domain into our structs.
        # Process the type hierarchy. Sort the types such that if X inherits from Y
        # then X is after Y in the list (topological sort).
        type_graph = {
            t: {t.parent} for t in pyperplan_types.values() if t.parent is not None
        }
        sorted_types = list(TopologicalSorter(type_graph).static_order())
        pyperplan_type_to_type: Dict[PyperplanType, Type] = {}
        for pyper_type in sorted_types:
            if pyper_type.parent is None:
                assert pyper_type.name == "object"
                parent = None
            else:
                parent = pyperplan_type_to_type[pyper_type.parent]
            new_type = Type(pyper_type.name, parent)
            pyperplan_type_to_type[pyper_type] = new_type
        # Handle case where the domain is untyped.
        # Pyperplan uses the object type by default.
        if not pyperplan_type_to_type:
            pyper_type = next(iter(pyperplan_types.values()))
            new_type = Type(pyper_type.name, parent=None)
            pyperplan_type_to_type[pyper_type] = new_type
        # Convert the predicates.
        predicate_name_to_predicate = {}
        for pyper_pred in pyperplan_predicates.values():
            name = pyper_pred.name
            pred_types = [pyperplan_type_to_type[t] for _, (t,) in pyper_pred.signature]
            predicate_name_to_predicate[name] = Predicate(name, pred_types)
        # Convert the operators.
        operators = set()
        for pyper_op in pyperplan_operators.values():
            name = pyper_op.name
            parameters = [
                Variable(n, pyperplan_type_to_type[t]) for n, (t,) in pyper_op.signature
            ]
            param_name_to_param = {p.name: p for p in parameters}
            preconditions = {
                LiftedAtom(
                    predicate_name_to_predicate[a.name],
                    [param_name_to_param[n] for n, _ in a.signature],
                )
                for a in pyper_op.precondition
            }
            add_effects = {
                LiftedAtom(
                    predicate_name_to_predicate[a.name],
                    [param_name_to_param[n] for n, _ in a.signature],
                )
                for a in pyper_op.effect.addlist
            }
            delete_effects = {
                LiftedAtom(
                    predicate_name_to_predicate[a.name],
                    [param_name_to_param[n] for n, _ in a.signature],
                )
                for a in pyper_op.effect.dellist
            }
            # Get action cost if available
            action_cost = action_costs.get(name.lower())
            strips_op = LiftedOperator(
                name,
                parameters,
                preconditions,
                add_effects,
                delete_effects,
                action_cost,
            )
            operators.add(strips_op)
        # Collect the final outputs.
        types = set(pyperplan_type_to_type.values())
        predicates = set(predicate_name_to_predicate.values())
        return PDDLDomain(domain_name, operators, predicates, types, uses_action_costs)

    @cached_property
    def _hash(self) -> int:
        return consistent_hash(str(self))

    @cached_property
    def _str(self) -> str:
        # Sort everything to ensure determinism.
        preds_lst = sorted(self.predicates)
        # Case 1: no type hierarchy.
        if all(t.parent is None for t in self.types):
            types_str = " ".join(t.name for t in sorted(self.types))
        # Case 2: type hierarchy.
        else:
            parent_to_children_types: Dict[Type, List[Type]] = {
                t: [] for t in self.types
            }
            for t in sorted(self.types):
                if t.parent:
                    parent_to_children_types[t.parent].append(t)
            types_str = ""
            for parent_type in sorted(parent_to_children_types):
                child_types = parent_to_children_types[parent_type]
                if not child_types:
                    # Special case: type has no children and also does not appear
                    # as a child of another type.
                    is_child_type = any(
                        parent_type in children
                        for children in parent_to_children_types.values()
                    )
                    if not is_child_type:
                        types_str += f"\n    {parent_type.name}"
                    # Otherwise, the type will appear as a child elsewhere.
                else:
                    child_type_str = " ".join(t.name for t in child_types)
                    types_str += f"\n    {child_type_str} - {parent_type.name}"
            # NOTE: for children with no parents, we need to explicitly add them to the
            # type list as lone types. Otherwise, pyperplan parsing fails.
            parentless_children = set(parent_to_children_types)
            for children in parent_to_children_types.values():
                parentless_children -= set(children)
            for parent_type in sorted(parentless_children):
                # NOTE: the "object" type is special in pyperplan. It's always included
                # implicitly in the domain. Adding it in the PDDL leads to headaches.
                if parent_type.name == "object":
                    continue
                types_str += f"\n    {parent_type.name}"
        ops_lst = sorted(self.operators)
        preds_str = "\n    ".join(pred.pddl_str for pred in preds_lst)
        ops_strs = "\n\n  ".join(op.pddl_str for op in ops_lst)
        # Build requirements
        requirements = ":typing"
        if self.uses_action_costs:
            requirements += " :action-costs"
        # Build functions section for action costs
        functions_section = ""
        if self.uses_action_costs:
            functions_section = "\n\n    (:functions\n    (total-cost) - number\n    )"
        return f"""(define (domain {self.name})
    (:requirements {requirements})
    (:types {types_str}){functions_section}

    (:predicates\n    {preds_str}
    )

    {ops_strs}
)
"""

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str


@dataclass(frozen=True)
class PDDLProblem:
    """A PDDL problem."""

    domain_name: str
    problem_name: str
    objects: Collection[Object]
    init_atoms: Collection[GroundAtom]
    goal: Collection[GroundAtom]
    uses_action_costs: bool = False

    @classmethod
    def parse(cls, pddl_problem_str: str, pddl_domain: PDDLDomain) -> PDDLProblem:
        """Parse a problem from a string."""
        # Let pyperplan do most of the heavy lifting.
        pddl_domain_str = str(pddl_domain)
        if pddl_domain.uses_action_costs:
            pddl_domain_str = _remove_action_costs_from_domain_str(pddl_domain_str)
            pddl_problem_str = _remove_action_costs_from_problem_str(pddl_problem_str)
        pyperplan_domain = _domain_str_to_pyperplan_domain(pddl_domain_str)
        # Now that we have the domain, parse the problem.
        lisp_iterator = parse_lisp_iterator(pddl_problem_str.split("\n"))
        problem_ast = parse_problem_def(lisp_iterator)
        problem_name = problem_ast.name
        visitor = TraversePDDLProblem(pyperplan_domain)
        problem_ast.accept(visitor)
        pyperplan_problem = visitor.get_problem()
        # Create the objects.
        type_name_to_type = {t.name: t for t in pddl_domain.types}
        object_name_to_obj = {
            o: Object(o, type_name_to_type[t.name])
            for o, t in pyperplan_problem.objects.items()
        }
        objects = set(object_name_to_obj.values())
        # Create the initial state.
        predicate_name_to_predicate = {p.name: p for p in pddl_domain.predicates}
        init_atoms = {
            GroundAtom(
                predicate_name_to_predicate[a.name],
                [object_name_to_obj[n] for n, _ in a.signature],
            )
            for a in pyperplan_problem.initial_state
        }
        # Create the goal.
        goal = {
            GroundAtom(
                predicate_name_to_predicate[a.name],
                [object_name_to_obj[n] for n, _ in a.signature],
            )
            for a in pyperplan_problem.goal
        }
        return PDDLProblem(
            pddl_domain.name,
            problem_name,
            objects,
            init_atoms,
            goal,
            pddl_domain.uses_action_costs,
        )

    @cached_property
    def _hash(self) -> int:
        return consistent_hash(str(self))

    @cached_property
    def _str(self) -> str:
        # Sort everything to ensure determinism.
        objects_lst = sorted(self.objects)
        init_atoms_lst = sorted(self.init_atoms)
        goal_lst = sorted(self.goal)
        objects_str = "\n    ".join(f"{o.name} - {o.type.name}" for o in objects_lst)
        init_str = "\n    ".join(atom.pddl_str for atom in init_atoms_lst)
        # Add initial cost state if using action costs
        if self.uses_action_costs:
            if init_str:
                init_str += "\n    "
            init_str += "(= (total-cost) 0)"
        goal_str = "\n    ".join(atom.pddl_str for atom in goal_lst)
        # Build metric section for action costs
        metric_section = ""
        if self.uses_action_costs:
            metric_section = "\n    (:metric minimize (total-cost))"
        return f"""(define (problem {self.problem_name}) (:domain {self.domain_name})
    (:objects\n    {objects_str}
    )
    (:init\n    {init_str}
    )
    (:goal (and {goal_str})){metric_section}
)
"""

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str
