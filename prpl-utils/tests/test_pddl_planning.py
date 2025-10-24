"""Tests for pddl_planning.py."""

import pytest

from prpl_utils.pddl_planning import run_pddl_planner


@pytest.mark.skip("Fast downward is not installed on CI.")
def test_pddl_planning_action_costs() -> None:
    """Test that planning with action costs works as expected.

    Note that pyperplan does not support action costs.
    """

    domain_str = """(define (domain costly)
    (:requirements :typing :action-costs)
    
    (:types 
        location - object
    )
    
    (:functions
        (total-cost) - number
    )

    (:predicates
        (at ?loc - location)
        (drivable ?from - location ?to - location)
    )

    (:action drive
        :parameters (?from - location ?to - location)
        :precondition (and (at ?from) (drivable ?from ?to))
        :effect (and 
            (at ?to)
            (not (at ?from))
            (increase (total-cost) 1)
        )
    )

    (:action fly
        :parameters (?from - location ?to - location)
        :precondition (and (at ?from))
        :effect (and 
            (at ?to)
            (not (at ?from))
            (increase (total-cost) 10)
        )
    )
)
"""

    problem_str = """(define (problem test-problem) (:domain costly)
    (:objects
        loc1 loc2 loc3 - location
    )
    (:init
        (at loc1)
        (drivable loc1 loc2)
        (drivable loc2 loc3)
        (= (total-cost) 0)
    )
    (:goal (at loc3))
    (:metric minimize (total-cost))
)
"""

    plan = run_pddl_planner(domain_str, problem_str, planner="fd-opt")
    assert plan == ["(drive loc1 loc2)", "(drive loc2 loc3)"]
