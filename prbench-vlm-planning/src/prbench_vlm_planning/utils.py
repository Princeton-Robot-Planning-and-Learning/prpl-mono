"""Utilities."""

import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Hashable,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from numpy.typing import NDArray
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from relational_structs.objects import Object, Type


def create_vlm_by_name(model_name: str):
    """Create a VLM instance using prpl_llm_utils."""
    # Create a cache directory in the current working directory
    cache_dir = Path("./vlm_cache")
    cache_dir.mkdir(exist_ok=True)
    cache = FilePretrainedLargeModelCache(cache_dir)

    try:
        return OpenAIModel(model_name, cache)
    except Exception as e:
        logging.exception("Failed to create VLM model")
        raise ValueError(f"Failed to create VLM model: {e}")


def parse_model_output_into_option_plan(
    model_prediction: str,
    objects: Collection[Object],
    types: Collection[Type],
    options: dict[str, LiftedParameterizedController],
    parse_continuous_params: bool,
) -> list[tuple[LiftedParameterizedController, Sequence[Object], Sequence[float]]]:
    """Assuming text for an option plan that is predicted as text by a large model,
    parse it into a sequence of ParameterizedOptions coupled with a list of objects and
    continuous parameters that will be used to ground the ParameterizedOption.

    We assume the model's output is such that each line is formatted as
    option_name(obj0:type0, obj1:type1,...)[continuous_param0,
    continuous_param1, ...].
    """
    option_plan: list[
        tuple[LiftedParameterizedController, Sequence[Object], Sequence[float]]
    ] = []
    # Setup dictionaries enabling us to easily map names to specific
    # Python objects during parsing.
    option_name_to_option = dict(options.items())
    type_name_to_type = {typ.name: typ for typ in types}
    obj_name_to_obj = {o.name: o for o in objects}
    options_str_list = model_prediction.split("\n")
    for option_str in options_str_list:
        logging.debug(f"Parsing option string: {option_str}")
        option_str_stripped = option_str.strip()
        option_name = option_str_stripped.split("(")[0]
        # Skip empty option strs.
        if not option_str:
            continue
        if option_name not in option_name_to_option.keys() or "(" not in option_str:
            logging.info(
                f"Line {option_str} output by model doesn't "
                "contain a valid option name. Terminating option plan "
                "parsing."
            )
            break
        if parse_continuous_params and "[" not in option_str:
            logging.info(
                f"Line {option_str} output by model doesn't contain a "
                "'[' and is thus improperly formatted."
            )
            break
        option = option_name_to_option[option_name]
        # Now that we have the option, we need to parse out the objects
        # along with specified types.
        try:
            start_index = option_str_stripped.index("(") + 1
            end_index = option_str_stripped.index(")", start_index)
        except ValueError:
            logging.info(f"Line {option_str} output by model is improperly formatted.")
            break
        typed_objects_str_list = option_str_stripped[start_index:end_index].split(",")
        objs_list = []
        continuous_params_list = []
        malformed = False
        for i, type_object_string in enumerate(typed_objects_str_list):
            object_type_str_list = type_object_string.strip().split(":")
            # We expect this list to be [object_name, type_name].
            if len(object_type_str_list) != 2:
                logging.info(
                    f"Line {option_str} output by model has a "
                    "malformed object-type list."
                )
                malformed = True
                break
            object_name = object_type_str_list[0]
            type_name = object_type_str_list[1]
            if object_name not in obj_name_to_obj.keys():
                logging.info(
                    f"Line {option_str} output by model has an " "invalid object name."
                )
                malformed = True
                break
            obj = obj_name_to_obj[object_name]
            # Check that the type of this object agrees
            # with what's expected given the ParameterizedOption.
            if type_name not in type_name_to_type:
                logging.info(
                    f"Line {option_str} output by model has an " "invalid type name."
                )
                malformed = True
                break
            try:
                if option.types[i] not in type_name_to_type[type_name].get_ancestors():
                    logging.info(
                        f"Line {option_str} output by model has an "
                        "invalid type that doesn't agree with the option"
                        f"{option}"
                    )
                    malformed = True
                    break
            except IndexError:
                # In this case, there's more supplied arguments than the
                # option has.
                logging.info(
                    f"Line {option_str} output by model has an "
                    "too many object arguments for option"
                    f"{option}"
                )
                malformed = True
                break
            objs_list.append(obj)
        # The types of the objects match, but we haven't yet checked if
        # all arguments of the option have an associated object.
        if len(objs_list) != len(option.types):
            malformed = True
        # Now, we attempt to parse out the continuous parameters.
        if parse_continuous_params:
            params_str_list = option_str_stripped.split("[")[1].strip("]").split(",")
            for i, continuous_params_str in enumerate(params_str_list):
                stripped_continuous_param_str = continuous_params_str.strip()
                if len(stripped_continuous_param_str) == 0:
                    continue
                try:
                    curr_cont_param = float(stripped_continuous_param_str)
                except ValueError:
                    logging.info(
                        f"Line {option_str} output by model has an "
                        "invalid continouous parameter that can't be"
                        "converted to a float."
                    )
                    malformed = True
                    break
                continuous_params_list.append(curr_cont_param)
            if option.params_space is None:
                logging.info(
                    f"Line {option_str} output by model has "
                    "continuous parameters but option has no params_space."
                )
                malformed = True
                break
            if len(continuous_params_list) != option.params_space.shape[0]:
                logging.info(
                    f"Line {option_str} output by model has "
                    "invalid continouous parameter(s) that don't "
                    f"agree with {option}{option.params_space}."
                )
                malformed = True
                break
        if not malformed:
            option_plan.append((option, objs_list, continuous_params_list))
    return option_plan


_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


def controller_and_param_plan_to_policy(
    controller_and_param_plan: list[
        tuple[GroundParameterizedController, Sequence[float]]
    ],
    max_horizon: int,
    observation_space: Any,
) -> Callable[[_O], _U]:
    """Convert a controller plan to a policy."""
    queue = list(controller_and_param_plan)

    def _controller_and_params_policy(
        obs: _O,  # pylint: disable=unused-argument
    ) -> tuple[GroundParameterizedController, Sequence[float]]:
        if not queue:
            raise Exception("Controller plan exhausted")
        controller, params = queue.pop(0)
        logging.debug(f"Executing controller: {controller}")
        return controller, params

    return option_policy_to_policy(
        _controller_and_params_policy, max_horizon, observation_space
    )


def option_policy_to_policy(
    option_policy: Callable[
        [_O], tuple[GroundParameterizedController, Sequence[float]]
    ],
    max_horizon: Optional[int],
    observation_space: Any,
) -> Callable[[Any], Any]:
    """Create a policy that executes the given option policy."""
    cur_option: Optional[GroundParameterizedController] = None
    num_cur_option_steps = 0

    def _policy(obs: Any) -> Any:
        nonlocal cur_option, num_cur_option_steps

        # Convert observation to ObjectCentricState
        state = observation_space.devectorize(cast(NDArray[np.float32], obs))

        # Check if we need a new controller
        if cur_option is None or cur_option.terminated():
            if max_horizon is not None and num_cur_option_steps >= max_horizon:
                raise Exception("Exceeded max controller steps.")

            # Get new controller from the option policy
            cur_option, params = option_policy(obs)
            if len(params) == 0:
                params = [0.0]
            # Initialize the controller with its parameters
            cur_option.reset(state, params)  # pylint: disable=protected-access
            num_cur_option_steps = 0
        else:
            # Let the controller observe the current state
            cur_option.observe(state)

        num_cur_option_steps += 1

        # Get action from controller
        action = cur_option.step()
        return action

    return _policy
