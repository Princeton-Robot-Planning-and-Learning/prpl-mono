# Tidybot 3D Pick-and-Place Skills for Bilevel Planning

This implementation provides tidybot pick-and-place skills for use with the bilevel planning framework. The skills are based on the original tidybot motion planner policy but adapted to work with the relational state representation and bilevel planning architecture.

## Overview

The implementation consists of three main components:

1. **Core Utilities** (`tidybot3d_utils.py`): Base classes and utilities for tidybot control
2. **Ground Skills** (`ground.py`): Specific pick-and-place skill implementations
3. **Integration Example** (`example_tidybot_usage.py`): Demonstration of how to use the skills

## Key Features

- **Pick Skill**: Navigate to an object, position the arm, and grasp the object
- **Place Skill**: Navigate to a target location, position the arm, and release the object
- **Combined Pick-and-Place**: Complete sequence of picking up an object and placing it at a target location
- **State Conversion**: Utilities to convert between ObjectCentricState and tidybot observation formats
- **MuJoCo Integration**: Compatible with the existing tidybot MuJoCo environment

## Architecture

### TidybotController (Base Class)

The `TidybotController` class provides the foundation for all tidybot skills:

```python
class TidybotController(GroundParameterizedController):
    """Base controller for Tidybot 3D manipulation tasks."""
    
    def reset(self, x: ObjectCentricState, params: tuple[float, ...] | float) -> None:
        """Reset the controller for a new episode."""
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the controller."""
    
    def observe(self, x: ObjectCentricState) -> None:
        """Update the controller with new observations."""
    
    def terminated(self) -> bool:
        """Check if the skill execution is terminated."""
```

### Skill Controllers

#### PickController

Implements the pick skill by:
1. Detecting objects in the environment
2. Planning base movement to reach the object
3. Executing arm manipulation sequence (approach, lower, grasp, lift)

```python
pick_controller = PickController(
    objects=[robot, cube],
    max_skill_horizon=100,
    custom_grasp=False,
)
```

#### PlaceController

Implements the place skill by:
1. Planning base movement to reach target location
2. Executing arm manipulation sequence (approach, lower, release, home)

```python
place_controller = PlaceController(
    objects=[robot, cube],
    target_x=1.0,
    target_y=0.0,
    max_skill_horizon=100,
)
```

#### PickAndPlaceController

Combines pick and place into a single skill:

```python
pick_place_controller = PickAndPlaceController(
    objects=[robot, cube],
    target_x=1.0,
    target_y=0.0,
    max_skill_horizon=200,
)
```

## Usage

### Basic Usage

```python
from prbench_bilevel_planning.env_models.tidybot3d import (
    PickController, PlaceController, TidybotStateConverter
)
from relational_structs import Object, Type
from relational_structs.utils import create_state_from_dict

# Create objects
robot_type = Type("robot", ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"])
cube_type = Type("cube", ["x", "y", "z"])

robot = Object("tidybot", robot_type)
cube = Object("cube1", cube_type)

# Create initial state
initial_state_data = {
    robot: {"x": 0.0, "y": 0.0, "theta": 0.0, "arm_x": 0.14, "arm_y": 0.0, "arm_z": 0.21, "gripper": 0.0},
    cube: {"x": 0.5, "y": 0.1, "z": 0.02},
}

# Create type features
type_features = {
    robot_type: ["x", "y", "theta", "arm_x", "arm_y", "arm_z", "gripper"],
    cube_type: ["x", "y", "z"],
}

# Create state using the proper function
state = create_state_from_dict(initial_state_data, type_features)

# Create and use pick controller
pick_controller = PickController([robot, cube])
pick_controller.reset(state, params=0.0)

# Execute skill
while not pick_controller.terminated():
    action = pick_controller.step()
    # Execute action in your environment
    # Update state and provide feedback to controller
    pick_controller.observe(updated_state)
```

### Integration with MuJoCo Environment

```python
from prbench.envs.tidybot.tidybot_mujoco_env import MujocoEnv

# Create MuJoCo environment
env = MujocoEnv(
    render_images=False,
    show_viewer=True,
    table_scene=False,  # Use ground scene
    cupboard_scene=False,
)

# Reset environment
env.reset()

# Execute skill actions in environment
while not controller.terminated():
    action = controller.step()
    env.step(action)
    
    # Get observations from environment
    obs = env.get_obs()
    
    # Convert to state and update controller
    # (You'll need to implement obs to state conversion)
    updated_state = convert_obs_to_state(obs)
    controller.observe(updated_state)
```

### State Conversion Utilities

The `TidybotStateConverter` class provides utilities for converting between different state representations:

```python
# Convert ObjectCentricState to tidybot observation format
obs = TidybotStateConverter.state_to_obs(state, robot)

# Convert tidybot observation to state updates
updates = TidybotStateConverter.obs_to_state_update(obs, robot)

# Create tidybot action with default values
action = create_tidybot_action(
    base_pose=np.array([0.5, 0.1, 0.0]),
    arm_pos=np.array([0.2, 0.0, 0.3]),
    gripper_pos=np.array([1.0]),
)
```

## Configuration Parameters

### Controller Parameters

- `max_skill_horizon`: Maximum number of steps before termination (default: 100)
- `ee_offset`: End-effector offset for IK solver (default: 0.12)
- `custom_grasp`: Enable custom grasping behavior (default: False)

### Manipulation Parameters

The controllers use the following parameters from the original tidybot implementation:

```python
# Base following parameters
LOOKAHEAD_DISTANCE = 0.3
POSITION_TOLERANCE = 0.005
GRASP_BASE_TOLERANCE = 0.002
PLACE_BASE_TOLERANCE = 0.02

# Object and target locations
PLACEMENT_X_OFFSET = 1.0
PLACEMENT_Y_OFFSET = 0.3
PLACEMENT_Z_OFFSET = 0.0

# Manipulation parameters
ROBOT_BASE_HEIGHT = 0.48
PICK_APPROACH_HEIGHT_OFFSET = 0.25
PICK_LOWER_DIST = 0.08
PICK_LIFT_DIST = 0.28
PLACE_APPROACH_HEIGHT_OFFSET = 0.10

# Grasping parameters
GRASP_SUCCESS_THRESHOLD = 0.7
GRASP_PROGRESS_THRESHOLD = 0.3
GRASP_TIMEOUT_S = 3.0
PLACE_SUCCESS_THRESHOLD = 0.2
```

## Running the Example

To run the demonstration:

```bash
cd prbench-bilevel-planning
python example_tidybot_usage.py
```

The example will:
1. Set up the environment (with or without MuJoCo visualization)
2. Demonstrate state conversion utilities
3. Show individual pick and place skills
4. Demonstrate the combined pick-and-place skill

## Integration with Bilevel Planning

The skills are designed to work with the bilevel planning framework through the `create_bilevel_planning_models` function:

```python
from prbench_bilevel_planning.env_models.tidybot3d.ground import create_bilevel_planning_models

# Create bilevel planning models
models = create_bilevel_planning_models(
    observation_space=env.observation_space,
    executable_space=env.action_space,
    max_skill_horizon=100,
    custom_grasp=False,
)
```

This creates a `SesameModels` structure containing the skill controllers that can be used by the bilevel planner.

## Dependencies

The implementation requires:

- `numpy`: For numerical computations
- `bilevel_planning`: For the bilevel planning framework
- `relational_structs`: For object-centric state representation
- `prbench.envs.tidybot` (optional): For MuJoCo environment integration

Optional dependencies for full functionality:
- `mujoco`: For physics simulation
- `gymnasium`: For environment interface

## Extending the Skills

To add new skills, create a new controller class that inherits from `TidybotController`:

```python
class MyCustomSkill(TidybotController):
    def __init__(self, objects, **kwargs):
        super().__init__(objects, **kwargs)
        # Custom initialization
    
    def _generate_plan(self, state: ObjectCentricState) -> List[Dict[str, Any]]:
        """Generate a plan for your custom skill."""
        plan = []
        # Implement your skill logic here
        return plan
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed and the Python path includes the project directory
2. **MuJoCo Setup**: If MuJoCo components are not available, the skills will still work without visualization
3. **State Conversion**: Ensure that your ObjectCentricState includes all required attributes for the robot and objects

### Debugging

Enable debug output by modifying the controllers to print intermediate states:

```python
# Add debug prints in the controller step method
def step(self) -> Dict[str, Any]:
    action = super().step()
    print(f"Generated action: {action}")
    return action
```

## Contributing

To contribute new skills or improvements:

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update this README with any new features

## License

This implementation is part of the prbench-bilevel-planning project and follows the same license terms. 