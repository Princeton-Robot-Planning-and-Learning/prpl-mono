"""CSP generation for the cooking environment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, get_args

import numpy as np
from gymnasium.spaces import Discrete, Text
from tomsutils.spaces import FunctionalSpace

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.envs.cooking.cooking_hidden_spec import (
    MealPreferenceModel,
)
from multitask_personalization.envs.cooking.cooking_meals import Meal
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec
from multitask_personalization.envs.cooking.cooking_structs import (
    AddIngredientCookingAction,
    CookingAction,
    CookingState,
    MovePotCookingAction,
    MultiCookingAction,
    ServeMealCookingAction,
    WaitCookingAction,
)
from multitask_personalization.envs.cooking.cooking_utils import (
    calculate_total_cooking_time,
)
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)
from multitask_personalization.utils import _NO_CHANGE, _NoChange


@dataclass(frozen=True)
class _IngredientCSPState:

    name: str
    is_used: bool
    quantity: float
    pot_id: int
    start_time: int
    pos: tuple[float, float]

    def calculate_final_temperature(
        self, scene_spec: CookingSceneSpec, total_time: int
    ) -> float:
        """Calculate the final ingredient temperature."""
        ingredient_cooking_time = total_time - self.start_time - 1
        if ingredient_cooking_time <= 0:
            return 0.0
        heat_rate = scene_spec.get_ingredient(self.name).heat_rate
        return ingredient_cooking_time * heat_rate

    def copy_with(
        self,
        is_used: bool | _NoChange = _NO_CHANGE,
        quantity: float | _NoChange = _NO_CHANGE,
        pot_id: int | _NoChange = _NO_CHANGE,
        start_time: int | _NoChange = _NO_CHANGE,
        pos: tuple[float, float] | _NoChange = _NO_CHANGE,
    ) -> _IngredientCSPState:
        """Create a new ingredient state."""
        return _IngredientCSPState(
            self.name,
            is_used=self.is_used if isinstance(is_used, _NoChange) else is_used,
            quantity=self.quantity if isinstance(quantity, _NoChange) else quantity,
            pot_id=self.pot_id if isinstance(pot_id, _NoChange) else pot_id,
            start_time=(
                self.start_time if isinstance(start_time, _NoChange) else start_time
            ),
            pos=self.pos if isinstance(pos, _NoChange) else pos,
        )


class CookingCSPGenerator(CSPGenerator[CookingState, CookingAction]):
    """Generates CSPs for the cooking environment."""

    def __init__(
        self,
        scene_spec: CookingSceneSpec,
        meal_model: MealPreferenceModel,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._scene_spec = scene_spec
        self._meal_model = meal_model

    def save(self, model_dir: Path) -> None:
        self._meal_model.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._meal_model.load(model_dir)

    def _generate_variables(
        self,
        obs: CookingState,
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:

        variable_space: FunctionalSpace = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, get_args(_IngredientCSPState)),
        )

        # One per ingredient with: is_used, quantity, pot_id, start_time.
        ingredients = sorted(obs.ingredients)
        ingredient_variables = [CSPVariable(i, variable_space) for i in ingredients]

        # One "global" variable for the meal name.
        meal_name = CSPVariable("meal-name", Text(10000000))

        # One "global" variable for the total amount of cooking time.
        cooking_time = CSPVariable("total-cooking-time", Discrete(10000000))

        variables = ingredient_variables + [meal_name, cooking_time]

        # Initialization.
        initialization: dict[CSPVariable, Any] = {
            v: _IngredientCSPState(v.name, False, 0, -1, 0, (-1, -1))
            for v in ingredient_variables
        }
        initialization[meal_name] = "<unknown>"
        initialization[cooking_time] = 0

        return variables, initialization

    def _generate_personal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        ingredient_variables = variables[:-2]
        meal_name_variable, cooking_time_variable = variables[-2:]

        # Final ingredients must comprise some meal that the user enjoys.
        def _user_enjoys_meal_logprob(
            meal_name: str,
            total_cooking_time: int,
            *ingredients: _IngredientCSPState,
        ) -> float:
            meal = self._derive_meal(meal_name, total_cooking_time, *ingredients)
            lp = self._meal_model.predict_enjoyment_logprob(meal)
            return lp

        user_enjoys_meal_constraint = LogProbCSPConstraint(
            "user_enjoys_meal_constraint",
            [meal_name_variable, cooking_time_variable] + ingredient_variables,
            _user_enjoys_meal_logprob,
            threshold=np.log(0.5) - 1e-3,
        )

        return [user_enjoys_meal_constraint]

    def _generate_nonpersonal_constraints(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> list[CSPConstraint]:

        constraints: list[CSPConstraint] = []
        ingredient_variables = variables[:-2]
        meal_name_variable, cooking_time_variable = variables[-2:]

        # Pots cannot overlap on the stove.
        def _pots_nonoverlapping(
            ingredient1: _IngredientCSPState, ingredient2: _IngredientCSPState
        ) -> bool:
            # Unused pots need not be checked.
            if not ingredient1.is_used or not ingredient2.is_used:
                return True
            pot1 = ingredient1.pot_id
            pot2 = ingredient2.pot_id
            r1 = self._scene_spec.pots[pot1].radius
            r2 = self._scene_spec.pots[pot2].radius
            dist = np.linalg.norm(np.array(ingredient1.pos) - np.array(ingredient2.pos))
            return bool(dist >= r1 + r2)

        for i, ingredient1 in enumerate(ingredient_variables[:-1]):
            for ingredient2 in ingredient_variables[i + 1 :]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-nonoverlapping",
                    [ingredient1, ingredient2],
                    _pots_nonoverlapping,
                )
                constraints.append(constraint)

        # Ingredients cannot be in the same pot.
        def _ingredients_different_pots(
            ingredient1: _IngredientCSPState, ingredient2: _IngredientCSPState
        ) -> bool:
            # Unused pots need not be checked.
            if not ingredient1.is_used or not ingredient2.is_used:
                return True
            return ingredient1.pot_id != ingredient2.pot_id

        for i, ingredient1 in enumerate(ingredient_variables[:-1]):
            for ingredient2 in ingredient_variables[i + 1 :]:
                constraint = FunctionalCSPConstraint(
                    f"{ingredient1.name}-{ingredient2.name}-different-pots",
                    [ingredient1, ingredient2],
                    _ingredients_different_pots,
                )
                constraints.append(constraint)

        # Used ingredients must be in some pot.
        def _used_ingredient_in_pot(
            ingredient: _IngredientCSPState,
        ) -> bool:
            return not ingredient.is_used or ingredient.pot_id != -1

        for ingredient in ingredient_variables:
            constraint = FunctionalCSPConstraint(
                f"{ingredient.name}-in-pot-if-used",
                [ingredient],
                _used_ingredient_in_pot,
            )
            constraints.append(constraint)

        # Ingredient quantity used must be not more than total available.
        def _ingredient_quantity_exists(ingredient: _IngredientCSPState) -> bool:
            available = obs.ingredients[ingredient.name].ingredient_unused_quantity
            return ingredient.quantity <= available

        for ingredient in ingredient_variables:
            constraint = FunctionalCSPConstraint(
                f"{ingredient.name}-quantity-exists",
                [ingredient],
                _ingredient_quantity_exists,
            )
            constraints.append(constraint)

        # The ingredients must comprise SOME known meal.
        def _meal_is_known(
            meal_name: str,
            total_cooking_time: int,
            *ingredients: _IngredientCSPState,
        ) -> bool:
            meal = self._derive_meal(meal_name, total_cooking_time, *ingredients)
            for meal_spec in self._scene_spec.universal_meal_specs:
                if meal_spec.check(meal):
                    return True
            return False

        meal_is_known_constraint = FunctionalCSPConstraint(
            "meal-is-known",
            [meal_name_variable, cooking_time_variable] + ingredient_variables,
            _meal_is_known,
        )
        constraints.append(meal_is_known_constraint)

        return constraints

    def _generate_exploit_cost(
        self,
        obs: CookingState,
        variables: list[CSPVariable],
    ) -> CSPCost | None:
        return None

    def _generate_samplers(
        self,
        obs: CookingState,
        csp: CSP,
    ) -> list[CSPSampler]:

        ingredient_variables = csp.variables[:-2]
        meal_name_variable, cooking_time_variable = csp.variables[-2:]

        # Sample ingredients by sampling a meal from the meal model.
        def _sample_meal(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            # Sample a meal.
            meal = self._meal_model.sample(rng)
            # First determine the total amount of cooking time needed.
            total_cooking_time = calculate_total_cooking_time(meal, self._scene_spec)
            new_sol: dict[CSPVariable, Any] = {
                meal_name_variable: meal.name,
                cooking_time_variable: total_cooking_time,
            }
            # Now determine the start times for individual ingredients.
            for v in ingredient_variables:
                # Don't change positions, pots, etc.
                old_ing_state = sol[v]
                assert isinstance(old_ing_state, _IngredientCSPState)
                if v.name in meal.ingredients:
                    # Determine the temperature and quantity necessary for the meal.
                    temp, quant = meal.ingredients[v.name]
                    # Determine the cooking start time from the temperature.
                    heat_rate = self._scene_spec.get_ingredient(v.name).heat_rate
                    # The plus 1 is to account for the "add" time.
                    cooking_duration = int(np.round(temp / heat_rate)) + 1
                    start_time = total_cooking_time - cooking_duration
                    ing_state = old_ing_state.copy_with(
                        is_used=True, quantity=quant, start_time=start_time
                    )
                else:
                    ing_state = old_ing_state.copy_with(is_used=False)
                new_sol[v] = ing_state
            return new_sol

        meal_sampler = FunctionalCSPSampler(_sample_meal, csp, set(csp.variables))

        # Sample pots and positions for any used ingredients.
        def _sample_pots(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            unused_pot_ids = list(range(len(obs.pots)))
            new_sol = {}
            for v in ingredient_variables:
                old_ing_state = sol[v]
                assert isinstance(old_ing_state, _IngredientCSPState)
                if old_ing_state.is_used:
                    pot_id = rng.choice(unused_pot_ids)
                    radius = self._scene_spec.pots[pot_id].radius
                    unused_pot_ids.remove(pot_id)
                    pot_pos_x = rng.uniform(
                        radius, self._scene_spec.stove_top_width - radius
                    )
                    pot_pos_y = rng.uniform(
                        radius, self._scene_spec.stove_top_height - radius
                    )
                    pot_pos = (pot_pos_x, pot_pos_y)
                else:
                    pot_id = -1
                    pot_pos = (-1, -1)
                ing_state = old_ing_state.copy_with(pot_id=pot_id, pos=pot_pos)
                new_sol[v] = ing_state
            return new_sol

        pot_position_sampler = FunctionalCSPSampler(
            _sample_pots, csp, set(csp.variables)
        )

        return [meal_sampler, pot_position_sampler]

    def _derive_meal(
        self,
        meal_name: str,
        total_cooking_time: int,
        *ingredients: _IngredientCSPState,
    ) -> Meal:
        meal_ingredients = {}
        for ing_state in ingredients:
            if not ing_state.is_used:
                continue
            temp = ing_state.calculate_final_temperature(
                self._scene_spec, total_cooking_time
            )
            quant = ing_state.quantity
            meal_ingredients[ing_state.name] = (temp, quant)
        return Meal(meal_name, meal_ingredients)

    def _generate_policy(
        self,
        obs: CookingState,
        csp_variables: Collection[CSPVariable],
    ) -> CSPPolicy:
        return _CookingCSPPolicy(csp_variables, seed=self._seed)

    def observe_transition(
        self,
        obs: CookingState,
        act: CookingAction,
        next_obs: CookingState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._disable_learning:
            self._update_meal_model(obs, act, next_obs)

    def _update_meal_model(
        self, obs: CookingState, act: CookingAction, next_obs: CookingState
    ) -> None:
        # lifelong learning, adapt to preference shift
        self._meal_model.adapt_to_preference_shift()
        if not isinstance(act, ServeMealCookingAction):
            return
        meal = obs.get_meal(act.meal_name)
        critiques = next_obs.critiques
        self._meal_model.update(meal, critiques)


class _CookingCSPPolicy(CSPPolicy[CookingState, CookingAction]):

    def __init__(
        self,
        csp_variables: Collection[CSPVariable],
        seed: int = 0,
    ) -> None:
        super().__init__(csp_variables, seed)
        self._current_plan: list[CookingAction] = []
        self._terminated = False

    def _get_plan(self, obs: CookingState) -> list[CookingAction]:
        plan: list[CookingAction] = []
        ing_states: dict[str, _IngredientCSPState] = {
            v: self._get_value(v) for v in obs.ingredients
        }
        meal_name = self._get_value("meal-name")
        total_cooking_time = self._get_value("total-cooking-time")

        # First move all the pots onto the stove.
        for ing in sorted(ing_states):
            ing_state = ing_states[ing]
            if not ing_state.is_used:
                continue
            action: CookingAction = MovePotCookingAction(
                ing_state.pot_id, ing_state.pos
            )
            plan.append(action)

        # Determine what ingredients should be added at what times.
        cooking_time_to_ingredients: dict[int, set[str]] = defaultdict(set)
        for ing in sorted(ing_states):
            ing_state = ing_states[ing]
            if not ing_state.is_used:
                continue
            cooking_time_to_ingredients[ing_state.start_time].add(ing)

        # Cook and wait.
        for t in range(total_cooking_time):
            ings_to_add = cooking_time_to_ingredients[t]
            if not ings_to_add:
                action = WaitCookingAction()
            else:
                inner_actions: list[CookingAction] = []
                for ing in sorted(ings_to_add):
                    ing_state = ing_states[ing]
                    inner_action = AddIngredientCookingAction(
                        pot_id=ing_state.pot_id,
                        ingredient=ing,
                        ingredient_quantity=ing_state.quantity,
                    )
                    inner_actions.append(inner_action)
                action = MultiCookingAction(inner_actions)
            plan.append(action)

        # Serve the meal.
        plan.append(ServeMealCookingAction(meal_name))

        return plan

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._current_plan = []
        self._terminated = False

    def step(self, obs: CookingState) -> CookingAction:
        if not self._current_plan:
            plan = self._get_plan(obs)
            assert plan is not None
            self._current_plan = plan
        action = self._current_plan.pop(0)
        self._terminated = isinstance(action, ServeMealCookingAction)
        return action

    def check_termination(self, obs: CookingState) -> bool:
        return self._terminated
