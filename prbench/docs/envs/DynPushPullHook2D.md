# DynPushPullHook2D

![random action GIF](assets/random_action_gifs/DynPushPullHook2D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 2D physics-based tool-use environment where a robot must use a hook to push/pull a target block onto a middle wall (goal surface). The target block is positioned in the upper region of the world, while the middle wall is located at the center. The robot must manipulate the hook to navigate the target block downward through obstacles.

The target block is initially surrounded by obstacle blocks.

The robot has a movable circular base and an extendable arm with gripper fingers. The hook is a kinematic object that can be grasped and used as a tool to indirectly manipulate the target block. All dynamic objects follow PyMunk physics including gravity, friction, and collisions.

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.


## Available Variants
The number of obstructions differs between environment variants. For example, DynPushPullHook2D-o0 has no obstructions, while DynPushPullHook2D-o5 has 5 obstructions.

- `prbench/DynPushPullHook2D-o0-v0` (o0)
- `prbench/DynPushPullHook2D-o1-v0` (o1)
- `prbench/DynPushPullHook2D-o5-v0` (o5)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/DynPushPullHook2D.gif)

## Example Demonstration
![demo GIF](assets/demo_gifs/DynPushPullHook2D-o5/DynPushPullHook2D-o5_seed2_1768427291.gif)

**Demo Stats**: Total Reward: -194.00, Success: Yes, Steps: 194

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | vx_base |
| 4 | robot | vy_base |
| 5 | robot | omega_base |
| 6 | robot | vx_arm |
| 7 | robot | vy_arm |
| 8 | robot | omega_arm |
| 9 | robot | vx_gripper_l |
| 10 | robot | vy_gripper_l |
| 11 | robot | omega_gripper_l |
| 12 | robot | vx_gripper_r |
| 13 | robot | vy_gripper_r |
| 14 | robot | omega_gripper_r |
| 15 | robot | static |
| 16 | robot | base_radius |
| 17 | robot | arm_joint |
| 18 | robot | arm_length |
| 19 | robot | gripper_base_width |
| 20 | robot | gripper_base_height |
| 21 | robot | finger_gap |
| 22 | robot | finger_height |
| 23 | robot | finger_width |
| 24 | hook | x |
| 25 | hook | y |
| 26 | hook | theta |
| 27 | hook | vx |
| 28 | hook | vy |
| 29 | hook | omega |
| 30 | hook | static |
| 31 | hook | held |
| 32 | hook | color_r |
| 33 | hook | color_g |
| 34 | hook | color_b |
| 35 | hook | z_order |
| 36 | hook | width |
| 37 | hook | length_side1 |
| 38 | hook | length_side2 |
| 39 | hook | mass |
| 40 | target_block | x |
| 41 | target_block | y |
| 42 | target_block | theta |
| 43 | target_block | vx |
| 44 | target_block | vy |
| 45 | target_block | omega |
| 46 | target_block | static |
| 47 | target_block | held |
| 48 | target_block | color_r |
| 49 | target_block | color_g |
| 50 | target_block | color_b |
| 51 | target_block | z_order |
| 52 | target_block | width |
| 53 | target_block | height |
| 54 | target_block | mass |
| 55 | obstruction0 | x |
| 56 | obstruction0 | y |
| 57 | obstruction0 | theta |
| 58 | obstruction0 | vx |
| 59 | obstruction0 | vy |
| 60 | obstruction0 | omega |
| 61 | obstruction0 | static |
| 62 | obstruction0 | held |
| 63 | obstruction0 | color_r |
| 64 | obstruction0 | color_g |
| 65 | obstruction0 | color_b |
| 66 | obstruction0 | z_order |
| 67 | obstruction0 | width |
| 68 | obstruction0 | height |
| 69 | obstruction0 | mass |


## Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.065 | 0.065 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | dgripper | Change in gripper gap (positive is open) | -0.020 | 0.020 |


## Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when the target block reaches the middle wall (goal surface).

## References
This is a dynamic version of PushPullHook2D.
