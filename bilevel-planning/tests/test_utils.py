"""Tests for utils.py."""

from relational_structs import LiftedOperator, Predicate, Type
from relational_structs.utils import all_ground_operators

from bilevel_planning.structs import RelationalAbstractState
from bilevel_planning.utils import RelationalAbstractSuccessorGenerator


def test_successor_generator_with_ground_operators():
    """Test that providing ground_operators skips grounding and uses them."""
    loc_type = Type("loc")
    at = Predicate("at", [loc_type])
    connected = Predicate("connected", [loc_type, loc_type])

    src = loc_type("?src")
    dst = loc_type("?dst")
    move = LiftedOperator(
        "move",
        [src, dst],
        preconditions={at([src]), connected([src, dst])},
        add_effects={at([dst])},
        delete_effects={at([src])},
    )

    loc_a = loc_type("a")
    loc_b = loc_type("b")
    loc_c = loc_type("c")
    objects = {loc_a, loc_b, loc_c}

    # Build a state where the robot is at A, with connections A->B and B->C.
    atoms = {
        at([loc_a]),
        connected([loc_a, loc_b]),
        connected([loc_b, loc_c]),
    }
    state = RelationalAbstractState(atoms, objects)

    # Without ground_operators: generates all groundings (including ones
    # we don't want, like move(a, c) which is inapplicable anyway).
    gen_default = RelationalAbstractSuccessorGenerator({move})
    default_results = list(gen_default(state))
    assert len(default_results) == 1
    _, next_state = default_results[0]
    assert at([loc_b]) in next_state.atoms
    assert at([loc_a]) not in next_state.atoms

    # With ground_operators: only supply a subset (just A->B).
    full_ground_ops = all_ground_operators({move}, objects)
    subset = {op for op in full_ground_ops if op.parameters == [loc_a, loc_b]}
    assert len(subset) == 1

    gen_custom = RelationalAbstractSuccessorGenerator(
        {move}, precomputed_ground_operators=subset
    )
    custom_results = list(gen_custom(state))
    # Same single applicable result.
    assert len(custom_results) == 1
    assert custom_results[0][1] == next_state

    # With ground_operators: supply an empty set to block all successors.
    gen_empty = RelationalAbstractSuccessorGenerator(
        {move}, precomputed_ground_operators=set()
    )
    assert not list(gen_empty(state))
