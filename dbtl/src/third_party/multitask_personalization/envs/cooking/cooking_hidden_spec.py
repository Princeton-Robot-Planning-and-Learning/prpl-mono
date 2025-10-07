"""Hidden specification for the cooking environment."""

from __future__ import annotations

import abc
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multitask_personalization.envs.cooking.cooking_meals import (
    IngredientSpec,
    Meal,
    MealSpec,
)
from multitask_personalization.envs.cooking.cooking_structs import (
    IngredientCritique,
    PreferenceShiftSpec,
)
from multitask_personalization.utils import Bounded1DClassifier


@dataclass(frozen=True)
class CookingHiddenSpec:
    """Hidden parameters for a cooking environment."""

    meal_preference_model: MealPreferenceModel


class MealPreferenceModel(abc.ABC):
    """A model of a user's meal preferences."""

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator) -> Meal:
        """Sample a meal that the user should enjoy."""

    @abc.abstractmethod
    def meal_is_known(self, meal_name: str) -> bool:
        """Whether the name of the meal is known."""

    @abc.abstractmethod
    def predict_enjoyment_logprob(self, meal: Meal) -> float:
        """Predict the log probability of enjoyment."""

    @abc.abstractmethod
    def get_feedback(self, meal: Meal) -> list[IngredientCritique]:
        """Critique all the ingredients in this meal."""

    @abc.abstractmethod
    def update(self, meal: Meal, critiques: list[IngredientCritique]) -> None:
        """Update the model from user feedback."""

    @abc.abstractmethod
    def save(self, model_dir: Path) -> None:
        """Save the current meal model."""

    @abc.abstractmethod
    def load(self, model_dir: Path) -> None:
        """Load the meal model."""

    @abc.abstractmethod
    def shift_preferences(self, rng: np.random.Generator, current_step: int) -> bool:
        """Shift the user's preferences.

        Returns:
            bool: True if a shift occurred, False otherwise
        """

    @abc.abstractmethod
    def sync_variables(self, other) -> None:
        """Sync the variables of this object with another object."""

    @abc.abstractmethod
    def adapt_to_preference_shift(self) -> None:
        """Adapt the model to a preference shift by potentially resetting
        memory."""


class MealSpecMealPreferenceModel(MealPreferenceModel):
    """An explicit list of a user's meal preferences."""

    def __init__(
        self,
        meal_specs: list[MealSpec],
        preference_shift_spec: PreferenceShiftSpec,
        lifelong_learning: dict | None = None,
    ) -> None:
        self._universal_meal_specs = {m.name: m for m in meal_specs}
        print(f"self._universal_meal_specs: {self._universal_meal_specs}")
        assert len(self._universal_meal_specs) == len(
            meal_specs
        ), "Meal names must be unique"
        # Initialize models for quantity and temperature for each meal ingredient.
        self._temperature_models: dict[str, dict[str, Bounded1DClassifier]] = {}
        self._quantity_models: dict[str, dict[str, Bounded1DClassifier]] = {}
        for meal_name, meal_spec in self._universal_meal_specs.items():
            self._temperature_models[meal_name] = {}
            self._quantity_models[meal_name] = {}
            for ing_spec in meal_spec.ingredients:
                temp_lo, temp_hi = ing_spec.temperature
                self._temperature_models[meal_name][ing_spec.name] = (
                    Bounded1DClassifier(temp_lo, temp_hi)
                )
                quant_lo, quant_hi = ing_spec.quantity
                self._quantity_models[meal_name][ing_spec.name] = Bounded1DClassifier(
                    quant_lo,
                    quant_hi,
                )
        # Lifelong learning setup.
        # If a shift occurs, all ingredients will shift at the same time.
        # The amount is sampled within a range.

        # min number of steps before preference shift
        self._min_shift_interval = preference_shift_spec.min_shift_interval
        # probability of a preference shift
        self._shift_prob = preference_shift_spec.shift_prob
        # Range of the shift factor.
        # Given old range [x-r, x+r], new range is [max(0, x*f-r), x*f+r]
        self._shift_factor_range = preference_shift_spec.shift_factor_range

        # Lifelong learning configuration
        self._lifelong_learning = lifelong_learning or {
            "enabled": False,
            "method": "periodic_reset",
            "reset_interval": 100,
        }
        # Counter for tracking number of update calls
        self._update_counter = 0

    def sync_variables(self, other):
        """Sync the variables of this object with another object."""
        assert isinstance(other, self.__class__)
        self._universal_meal_specs = copy.deepcopy(
            other._universal_meal_specs  # pylint: disable=protected-access
        )
        # the models don't matter because the model update happens
        # in the approach copies of hidden spec

    def shift_preferences(self, rng: np.random.Generator, current_step: int) -> bool:
        """Shift the user's preferences.

        Returns:
            bool: True if a shift occurred, False otherwise
        """
        if (
            current_step > 0
            and current_step % self._min_shift_interval == 0
            and rng.uniform() < self._shift_prob
        ):
            # print the ingredient preferences before shift
            logging.info("Ingredient preferences before shift:")
            for meal_name, meal_spec in self._universal_meal_specs.items():
                logging.info(f"Meal: {meal_name}")
                for ing_spec in meal_spec.ingredients:
                    logging.info(
                        f"Ingredient: {ing_spec.name}, "
                        + f"Temperature: {ing_spec.temperature}, "
                        + f"Quantity: {ing_spec.quantity}"
                    )
            ing_shift_factors = {}
            for meal_name, meal_spec in self._universal_meal_specs.items():
                shifted_ing_specs = []
                for ing_spec in meal_spec.ingredients:
                    # Sample shift factors.
                    if ing_spec.name not in ing_shift_factors:
                        ing_shift_factors[ing_spec.name] = (
                            rng.uniform(*self._shift_factor_range),
                            rng.uniform(*self._shift_factor_range),
                        )
                    temp_shift_factor, quant_shift_factor = ing_shift_factors[
                        ing_spec.name
                    ]
                    # Shift temperature and quantity ranges.
                    temp_lo, temp_hi = ing_spec.temperature
                    temp_mean, temp_radius = (temp_lo + temp_hi) / 2, (
                        temp_hi - temp_lo
                    ) / 2
                    temp_shifted = (
                        max(0, temp_mean * temp_shift_factor - temp_radius),
                        temp_mean * temp_shift_factor + temp_radius,
                    )

                    quant_lo, quant_hi = ing_spec.quantity
                    quant_mean, quant_radius = (quant_lo + quant_hi) / 2, (
                        quant_hi - quant_lo
                    ) / 2
                    quant_shifted = (
                        max(0, quant_mean * quant_shift_factor - quant_radius),
                        quant_mean * quant_shift_factor + quant_radius,
                    )

                    shifted_ing_spec = IngredientSpec(
                        ing_spec.name, temperature=temp_shifted, quantity=quant_shifted
                    )
                    shifted_ing_specs.append(shifted_ing_spec)
                self._universal_meal_specs[meal_name] = MealSpec(
                    meal_name, shifted_ing_specs
                )

            # print the ingredient preferences after shift
            logging.info("Ingredient preferences after shift:")
            for meal_name, meal_spec in self._universal_meal_specs.items():
                logging.info(f"Meal: {meal_name}")
                for ing_spec in meal_spec.ingredients:
                    logging.info(
                        f"Ingredient: {ing_spec.name}, "
                        + f"Temperature: {ing_spec.temperature}, "
                        + f"Quantity: {ing_spec.quantity}"
                    )

            # log ing_shift_factors
            logging.info(
                f"Shifted preferences at step {current_step} with "
                + f"factors {ing_shift_factors}"
            )

            return True
        return False

    def adapt_to_preference_shift(self) -> None:
        """Adapt the model to a preference shift by potentially resetting
        memory."""
        self._update_counter += 1
        if self._lifelong_learning["enabled"]:
            # logging.info(f"Total update calls: {self._update_counter}")
            if (
                self._lifelong_learning["method"] == "periodic_reset"
                and self._update_counter > 1
                and (self._update_counter - 1)
                % self._lifelong_learning["reset_interval"]
                == 0
            ):
                logging.info(f"Resetting memory after {self._update_counter} updates")
                # reset for all meals and all ingredients
                self._temperature_models = {}
                self._quantity_models = {}
                for meal_name, meal_spec in self._universal_meal_specs.items():
                    self._temperature_models[meal_name] = {}
                    self._quantity_models[meal_name] = {}
                    for ing_spec in meal_spec.ingredients:
                        temp_lo, temp_hi = ing_spec.temperature
                        self._temperature_models[meal_name][ing_spec.name] = (
                            Bounded1DClassifier(temp_lo, temp_hi)
                        )
                        quant_lo, quant_hi = ing_spec.quantity
                        self._quantity_models[meal_name][ing_spec.name] = (
                            Bounded1DClassifier(quant_lo, quant_hi)
                        )

    def sample(self, rng: np.random.Generator) -> Meal:
        meal_spec_idx = rng.choice(len(self._universal_meal_specs))
        ordered_meal_specs = [
            self._universal_meal_specs[m] for m in sorted(self._universal_meal_specs)
        ]
        meal_spec = ordered_meal_specs[meal_spec_idx]
        return meal_spec.sample(rng)

    def meal_is_known(self, meal_name: str) -> bool:
        return meal_name in self._universal_meal_specs

    def predict_enjoyment_logprob(self, meal: Meal) -> float:
        if not meal.name in self._universal_meal_specs:
            return -np.inf
        # Accumulate estimates for the whole meal.
        meal_spec = self._universal_meal_specs[meal.name]
        total_log_prob = 0.0
        for ing_spec in meal_spec.ingredients:
            # If the ingredient is missing, fail.
            if ing_spec.name not in meal.ingredients:
                return -np.inf
            temp, quant = meal.ingredients[ing_spec.name]
            # Consider temperature.
            temperature_model = self._temperature_models[meal.name][ing_spec.name]
            temperature_log_prob = np.log(temperature_model.predict_proba([temp])[0])
            total_log_prob += temperature_log_prob
            # Consider quantity.
            quantity_model = self._quantity_models[meal.name][ing_spec.name]
            quantity_log_prob = np.log(quantity_model.predict_proba([quant])[0])
            total_log_prob += quantity_log_prob
        return total_log_prob

    def get_feedback(self, meal: Meal) -> list[IngredientCritique]:
        meal_spec = self._universal_meal_specs[meal.name]
        critiques: list[IngredientCritique] = []
        for ing_spec in meal_spec.ingredients:
            temperature_feedback = "good"
            quantity_feedback = "good"
            missing = False
            if ing_spec.name not in meal.ingredients:
                missing = True
            else:
                temp, quant = meal.ingredients[ing_spec.name]
                if temp < ing_spec.temperature[0]:
                    temperature_feedback = "hotter"
                elif temp > ing_spec.temperature[1]:
                    temperature_feedback = "colder"
                if quant < ing_spec.quantity[0]:
                    quantity_feedback = "more"
                elif quant > ing_spec.quantity[1]:
                    quantity_feedback = "less"
            critique = IngredientCritique(
                ing_spec.name,
                temperature_feedback,
                quantity_feedback,
                missing=missing,
            )
            critiques.append(critique)

        return critiques

    def update(self, meal: Meal, critiques: list[IngredientCritique]) -> None:
        for critique in critiques:
            assert not critique.missing
            meal_temp, meal_quant = meal.ingredients[critique.ingredient]
            temperature_label = critique.hotter_or_colder == "good"
            quantity_label = critique.more_or_less == "good"
            temperature_model = self._temperature_models[meal.name][critique.ingredient]
            quantity_model = self._quantity_models[meal.name][critique.ingredient]
            temperature_model.fit_incremental([meal_temp], [temperature_label])
            quantity_model.fit_incremental([meal_quant], [quantity_label])

    def save(self, model_dir: Path) -> None:
        # Create a dictionary of all the model parameters for readability.
        # Note that we need to save the data along with the model for full
        # saving and loading.
        model_parameters: dict = {}
        for meal_name, meal_spec in self._universal_meal_specs.items():
            model_parameters[meal_name] = {}
            for ing_spec in meal_spec.ingredients:
                ing_name = ing_spec.name
                temperature_model = self._temperature_models[meal_name][ing_name]
                quantity_model = self._quantity_models[meal_spec.name][ing_name]
                model_parameters[meal_name][ing_name] = {
                    "temperature": temperature_model.get_save_state(),
                    "quantity": quantity_model.get_save_state(),
                }
        filepath = model_dir / "meal_preferences.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_parameters, f)

    def load(self, model_dir: Path) -> None:
        filepath = model_dir / "meal_preferences.json"
        with open(filepath, "r", encoding="utf-8") as f:
            model_parameters = json.load(f)
        for meal_name, meal_spec in self._universal_meal_specs.items():
            for ing_spec in meal_spec.ingredients:
                ing_name = ing_spec.name
                ing_parameters = model_parameters[meal_name][ing_name]
                temperature_model = self._temperature_models[meal_name][ing_name]
                quantity_model = self._quantity_models[meal_spec.name][ing_name]
                for model, name in [
                    (temperature_model, "temperature"),
                    (quantity_model, "quantity"),
                ]:
                    model.load_from_state(ing_parameters[name])
