"""Utilities for the cooking environment."""

import numpy as np

from multitask_personalization.envs.cooking.cooking_meals import Meal
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec


def calculate_total_cooking_time(meal: Meal, scene_spec: CookingSceneSpec) -> int:
    """Calculate the total amount of time needed to cook this meal."""
    total_time = 0
    for ingredient, (temp, _) in meal.ingredients.items():
        heat_rate = scene_spec.get_ingredient(ingredient).heat_rate
        # The plus 1 is to account for the one "add" step.
        ingredient_time = int(np.round(temp / heat_rate)) + 1
        total_time = max(total_time, ingredient_time)
    return total_time
