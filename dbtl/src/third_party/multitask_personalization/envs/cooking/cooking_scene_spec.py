"""Scene specification for the cooking environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from multitask_personalization.envs.cooking.cooking_meals import (
    DEFAULT_MEAL_SPECS,
    MealSpec,
)
from multitask_personalization.envs.cooking.cooking_structs import (
    DEFAULT_PREFERENCE_SHIFT_SPEC,
    PreferenceShiftSpec,
)
from multitask_personalization.structs import PublicSceneSpec


@dataclass(frozen=True)
class CookingPot:
    """A pot in a cooking environment."""

    radius: float
    position: tuple[float, float] | None = None


@dataclass(frozen=True)
class CookingIngredient:
    """An ingredient in a cooking environment."""

    name: str
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)  # rendering
    respawn_quantity_bounds: tuple[float, float] = (10.0, 20.0)  # more than enough
    heat_rate: float = 0.1  # delta temperature during cooking
    cool_rate: float = 0.1  # delta temperature when not cooking (min = 0.0)


@dataclass(frozen=True)
class CookingSceneSpec(PublicSceneSpec):
    """Public parameters that define a cooking environment scene."""

    # A list of all known meal specs.
    universal_meal_specs: list[MealSpec] = field(
        default_factory=lambda: DEFAULT_MEAL_SPECS
    )

    # Preference shift parameters.
    preference_shift_spec: PreferenceShiftSpec = field(
        default=DEFAULT_PREFERENCE_SHIFT_SPEC
    )

    # The "stove top", a 2D rectangle. This is the only space in the env.
    stove_top_width: float = 10.0
    stove_top_height: float = 10.0

    # Characteristics of available pots.
    pots: list[CookingPot] = field(
        default_factory=lambda: [
            CookingPot(radius=0.5, position=None),
            CookingPot(radius=1.0, position=None),
            CookingPot(radius=0.5, position=None),
            CookingPot(radius=0.5, position=None),
        ]
    )

    # Characteristics of available ingredients.
    ingredients: list[CookingIngredient] = field(
        default_factory=lambda: [
            CookingIngredient(
                name="salt",
                color=(0.9, 0.9, 0.9),
            ),
            CookingIngredient(
                name="pepper",
                color=(0.0, 0.0, 0.0),
            ),
            CookingIngredient(
                name="sugar",
                color=(0.5, 0.0, 0.0),
            ),
            CookingIngredient(
                name="flour",
                color=(0.0, 0.5, 0.0),
            ),
        ]
    )

    # Rendering.
    render_figscale: float = 5
    render_padding: float = 0.1

    def get_ingredient(self, name: str) -> CookingIngredient:
        """Look up an ingredient by name."""
        for ingredient in self.ingredients:
            if name == ingredient.name:
                return ingredient
        raise ValueError(f"Unknown ingredient: {name}")
