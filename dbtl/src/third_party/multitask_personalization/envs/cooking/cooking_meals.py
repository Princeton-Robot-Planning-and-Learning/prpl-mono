"""Meal data structures and functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MealSpec:
    """A specification of one of the meals that the user enjoys."""

    # The unique name of the meal, e.g., "huevos rancheros".
    name: str

    # Ingredient name, temperature bounds, quantity bounds.
    ingredients: list[IngredientSpec]

    def __post_init__(self) -> None:
        for ing_spec in self.ingredients:
            assert 0 <= ing_spec.temperature[0] < ing_spec.temperature[1]
            assert 0 <= ing_spec.quantity[0] < ing_spec.quantity[1]

    def sample(self, rng: np.random.Generator) -> Meal:
        """Sample a meal."""
        ingredients = {}
        for ing_spec in self.ingredients:
            temp = rng.uniform(*ing_spec.temperature)
            quant = rng.uniform(*ing_spec.quantity)
            ingredients[ing_spec.name] = (temp, quant)
        return Meal(self.name, ingredients)

    def check(self, meal: Meal) -> bool:
        """Check if a meal fits the preferences."""
        for ing_spec in self.ingredients:
            if ing_spec.name not in meal.ingredients:
                return False
            temp, quant = meal.ingredients[ing_spec.name]
            if not ing_spec.temperature[0] <= temp <= ing_spec.temperature[1]:
                return False
            if not ing_spec.quantity[0] <= quant <= ing_spec.quantity[1]:
                return False
        return True


@dataclass(frozen=True)
class IngredientSpec:
    """An individual ingredient specification for a meal specification."""

    # The unique name of the ingredient.
    name: str

    # Lower and upper bounds on temperature.
    temperature: tuple[float, float]

    # Lower and upper bound on quantity.
    quantity: tuple[float, float]


@dataclass(frozen=True)
class Meal:
    """A specific number of ingredients."""

    # From the meal spec.
    name: str

    # Maps ingredient names to temperature and quantity.
    ingredients: dict[str, tuple[float, float]]


DEFAULT_MEAL_SPECS = [
    # MealSpec(
    #     "seasoning",
    #     [
    #         IngredientSpec("salt", temperature=(2.5, 3.5), quantity=(0.9, 1.1)),
    #         IngredientSpec("pepper", temperature=(2.5, 3.5), quantity=(0.9, 1.1)),
    #     ],
    # )
    MealSpec(
        "seasoning",
        [
            IngredientSpec("salt", temperature=(2.0, 4.0), quantity=(0.7, 1.3)),
            IngredientSpec("pepper", temperature=(2.0, 4.0), quantity=(0.7, 1.3)),
        ],
    )
]
