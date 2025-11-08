"""Tests for action costs in PDDL domains and problems."""

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    Type,
    Variable,
)


def test_operator_with_cost() -> None:
    """Test that operators can include action costs."""
    # Define types and predicates
    location_type = Type("location")
    robot_type = Type("robot")
    at_predicate = Predicate("at", [robot_type, location_type])

    # Define variables
    robot_var = Variable("?robot", robot_type)
    from_var = Variable("?from", location_type)
    to_var = Variable("?to", location_type)

    # Create operator with cost
    move_op = LiftedOperator(
        name="move",
        parameters=[robot_var, from_var, to_var],
        preconditions={LiftedAtom(at_predicate, [robot_var, from_var])},
        add_effects={LiftedAtom(at_predicate, [robot_var, to_var])},
        delete_effects={LiftedAtom(at_predicate, [robot_var, from_var])},
        cost=5,
    )

    # Check that cost is stored correctly
    assert move_op.cost == 5

    # Check PDDL string includes cost
    pddl_str = move_op.pddl_str
    assert "(increase (total-cost) 5)" in pddl_str

    # Test operator without cost
    move_op_no_cost = LiftedOperator(
        name="move",
        parameters=[robot_var, from_var, to_var],
        preconditions={LiftedAtom(at_predicate, [robot_var, from_var])},
        add_effects={LiftedAtom(at_predicate, [robot_var, to_var])},
        delete_effects={LiftedAtom(at_predicate, [robot_var, from_var])},
    )

    assert move_op_no_cost.cost is None
    assert "(increase (total-cost)" not in move_op_no_cost.pddl_str


def test_ground_operator_with_cost() -> None:
    """Test that ground operators preserve cost information."""
    # Define types, predicates, and objects
    location_type = Type("location")
    robot_type = Type("robot")
    at_predicate = Predicate("at", [robot_type, location_type])

    robot_var = Variable("?robot", robot_type)
    from_var = Variable("?from", location_type)
    to_var = Variable("?to", location_type)

    robot_obj = Object("robot1", robot_type)
    loc1_obj = Object("loc1", location_type)
    loc2_obj = Object("loc2", location_type)

    # Create lifted operator with cost
    move_op = LiftedOperator(
        name="move",
        parameters=[robot_var, from_var, to_var],
        preconditions={LiftedAtom(at_predicate, [robot_var, from_var])},
        add_effects={LiftedAtom(at_predicate, [robot_var, to_var])},
        delete_effects={LiftedAtom(at_predicate, [robot_var, from_var])},
        cost=3,
    )

    # Ground the operator
    ground_op = move_op.ground((robot_obj, loc1_obj, loc2_obj))

    # Check that cost is preserved
    assert ground_op.cost == 3
    assert "(increase (total-cost) 3)" in ground_op.pddl_str


def test_pddl_domain_with_action_costs() -> None:
    """Test parsing and creation of PDDL domains with action costs."""
    domain_str = """(define (domain logistics)
    (:requirements :typing :action-costs)
    (:types 
        location vehicle package - object
    )
    
    (:functions
        (total-cost) - number
    )

    (:predicates
        (at ?obj - object ?loc - location)
        (in ?pkg - package ?veh - vehicle)
    )

    (:action drive
        :parameters (?v - vehicle ?from - location ?to - location)
        :precondition (at ?v ?from)
        :effect (and 
            (at ?v ?to)
            (not (at ?v ?from))
                        (increase (total-cost) 2)
        )
    )

    (:action load
        :parameters (?pkg - package ?veh - vehicle ?loc - location)
        :precondition (and (at ?pkg ?loc) (at ?veh ?loc))
        :effect (and 
            (in ?pkg ?veh)
            (not (at ?pkg ?loc))
            (increase (total-cost) 1)
        )
    )
)
"""

    # Parse the domain
    domain = PDDLDomain.parse(domain_str)

    # Check that it recognizes action costs are used
    assert domain.uses_action_costs is True

    # Check that operators have correct costs
    operator_costs = {op.name: op.cost for op in domain.operators}
    assert operator_costs["drive"] == 2
    assert operator_costs["load"] == 1

    # Check that the domain string includes required elements
    domain_str_output = str(domain)
    assert ":action-costs" in domain_str_output
    assert "(:functions" in domain_str_output
    assert "(total-cost)" in domain_str_output
    assert "(increase (total-cost) 2)" in domain_str_output
    assert "(increase (total-cost) 1)" in domain_str_output


def test_pddl_domain_without_action_costs() -> None:
    """Test that domains without action costs work as before."""
    domain_str = """(define (domain simple)
    (:requirements :typing)
    (:types location robot)
    (:predicates
        (at ?r - robot ?l - location)
    )
    (:action move
        :parameters (?r - robot ?from - location ?to - location)
        :precondition (at ?r ?from)
        :effect (and (at ?r ?to) (not (at ?r ?from)))
    )
)
"""

    # Parse the domain
    domain = PDDLDomain.parse(domain_str)

    # Check that it doesn't use action costs
    assert domain.uses_action_costs is False

    # Check that operators don't have costs
    for op in domain.operators:
        assert op.cost is None

    # Check that domain string doesn't include cost-related elements
    domain_str_output = str(domain)
    assert ":action-costs" not in domain_str_output
    assert "(:functions" not in domain_str_output
    assert "(increase (total-cost)" not in domain_str_output


def test_pddl_problem_with_action_costs() -> None:
    """Test PDDL problems with action costs."""
    # First create a domain with action costs
    domain_str = """(define (domain test-domain)
    (:requirements :typing :action-costs)
    (:types location robot)
    (:functions (total-cost) - number)
    (:predicates (at ?r - robot ?l - location))
    (:action move
        :parameters (?r - robot ?from - location ?to - location)
        :precondition (at ?r ?from)
        :effect (and (at ?r ?to) (not (at ?r ?from)) (increase (total-cost) 1))
    )
)
"""

    domain = PDDLDomain.parse(domain_str)

    # Create a problem
    problem_str = """(define (problem test-problem) (:domain test-domain)
    (:objects
        robot1 - robot
        loc1 loc2 - location
    )
    (:init
        (at robot1 loc1)
    )
    (:goal (at robot1 loc2))
)
"""

    problem = PDDLProblem.parse(problem_str, domain)

    # Check that problem uses action costs
    assert problem.uses_action_costs is True

    # Check that problem string includes cost-related elements
    problem_str_output = str(problem)
    assert "(= (total-cost) 0)" in problem_str_output
    assert "(:metric minimize (total-cost))" in problem_str_output


def test_pddl_problem_without_action_costs() -> None:
    """Test that problems without action costs work as before."""
    domain_str = """(define (domain simple)
    (:requirements :typing)
    (:types location robot)
    (:predicates (at ?r - robot ?l - location))
    (:action move
        :parameters (?r - robot ?from - location ?to - location)
        :precondition (at ?r ?from)
        :effect (and (at ?r ?to) (not (at ?r ?from)))
    )
)
"""

    domain = PDDLDomain.parse(domain_str)

    problem_str = """(define (problem simple-problem) (:domain simple)
    (:objects
        robot1 - robot
        loc1 loc2 - location
    )
    (:init
        (at robot1 loc1)
    )
    (:goal (at robot1 loc2))
)
"""

    problem = PDDLProblem.parse(problem_str, domain)

    # Check that problem doesn't use action costs
    assert problem.uses_action_costs is False

    # Check that problem string doesn't include cost elements
    problem_str_output = str(problem)
    assert "(= (total-cost) 0)" not in problem_str_output
    assert "(:metric minimize (total-cost))" not in problem_str_output


def test_mixed_cost_operators() -> None:
    """Test domain with some operators having costs and others not."""
    domain_str = """(define (domain mixed-costs)
    (:requirements :typing :action-costs)
    (:types location robot)
    (:functions (total-cost) - number)
    (:predicates 
        (at ?r - robot ?l - location)
        (charged ?r - robot)
    )
    (:action move
        :parameters (?r - robot ?from - location ?to - location)
        :precondition (at ?r ?from)
        :effect (and (at ?r ?to) (not (at ?r ?from)) (increase (total-cost) 2))
    )
    (:action charge
        :parameters (?r - robot ?l - location)
        :precondition (at ?r ?l)
        :effect (charged ?r)
    )
)
"""

    domain = PDDLDomain.parse(domain_str)

    # Check that domain uses action costs
    assert domain.uses_action_costs is True

    # Check operator costs
    operator_costs = {op.name: op.cost for op in domain.operators}
    assert operator_costs["move"] == 2
    assert operator_costs["charge"] is None  # No cost specified

    # Check domain string
    domain_str_output = str(domain)
    move_action_in_output = (
        "move" in domain_str_output and "(increase (total-cost) 2)" in domain_str_output
    )
    charge_action_in_output = "charge" in domain_str_output
    assert move_action_in_output
    assert charge_action_in_output
    # Charge action should not have cost increase
    charge_section_start = domain_str_output.find("(:action charge")
    charge_section_end = domain_str_output.find(
        ")",
        charge_section_start + domain_str_output[charge_section_start:].find(":effect"),
    )
    charge_section = domain_str_output[charge_section_start:charge_section_end]
    assert "(increase (total-cost)" not in charge_section


def test_cost_extraction_edge_cases() -> None:
    """Test edge cases in cost extraction."""
    # Test with different whitespace and formatting
    domain_str_spaces = """(define (domain spaces)
    (:requirements :action-costs)
    (:functions (total-cost))
    (:predicates (p))
    (:action test
        :parameters ()
        :precondition (p)
        :effect (and (not (p)) (increase    (  total-cost  )   3   ))
    )
)
"""

    domain = PDDLDomain.parse(domain_str_spaces)
    assert domain.uses_action_costs is True
    op = next(iter(domain.operators))
    assert op.cost == 3

    # Test with integer costs
    domain_str_int = """(define (domain int-costs)
    (:requirements :action-costs)
    (:functions (total-cost))
    (:predicates (p))
    (:action test
        :parameters ()
        :precondition (p)
        :effect (and (not (p)) (increase (total-cost) 5))
    )
)
"""

    domain = PDDLDomain.parse(domain_str_int)
    op = next(iter(domain.operators))
    assert op.cost == 5


def test_problem_parsing_with_total_cost_and_metric() -> None:
    """Ensure PDDLProblem.parse handles problems that include total-cost and
    metric."""
    domain_str = """(define (domain test-domain)
    (:requirements :typing :action-costs)
    (:types location robot)
    (:functions (total-cost) - number)
    (:predicates (at ?r - robot ?l - location))
    (:action move
        :parameters (?r - robot ?from - location ?to - location)
        :precondition (at ?r ?from)
        :effect (and (at ?r ?to) (not (at ?r ?from)) (increase (total-cost) 1))
    )
)
"""

    domain = PDDLDomain.parse(domain_str)

    problem_str = """(define (problem test-problem) (:domain test-domain)
    (:objects
        robot1 - robot
        loc1 loc2 - location
    )
    (:init
        (at robot1 loc1)
        (= (total-cost) 0)
    )
    (:goal (at robot1 loc2))
    (:metric minimize (total-cost))
)
"""

    problem = PDDLProblem.parse(problem_str, domain)
    assert problem.uses_action_costs is True
