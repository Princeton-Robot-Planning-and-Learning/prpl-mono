# DynPushT

![random action GIF](assets/random_action_gifs/DynPushT.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 2D physics-based environment where the goal is to push a T-shaped block to match a goal pose using a simple dot robot (kinematic circle) with PyMunk physics simulation.

The T-shaped block must be positioned within small position and orientation thresholds of the goal.


## Available Variants
This environment has only one variant.

- `prbench/DynPushT-t1-v0` (t1)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/DynPushT.gif)

## Example Demonstration
![demo GIF](assets/demo_gifs/DynPushT-t1/DynPushT-t1.gif)

**Demo Stats**: Total Reward: -576.00, Success: Yes, Steps: 576

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | tblock | x |
| 1 | tblock | y |
| 2 | tblock | theta |
| 3 | tblock | vx |
| 4 | tblock | vy |
| 5 | tblock | omega |
| 6 | tblock | static |
| 7 | tblock | held |
| 8 | tblock | color_r |
| 9 | tblock | color_g |
| 10 | tblock | color_b |
| 11 | tblock | z_order |
| 12 | tblock | width |
| 13 | tblock | length_horizontal |
| 14 | tblock | length_vertical |
| 15 | tblock | mass |
| 16 | robot | x |
| 17 | robot | y |
| 18 | robot | theta |
| 19 | robot | vx |
| 20 | robot | vy |
| 21 | robot | omega |
| 22 | robot | static |
| 23 | robot | held |
| 24 | robot | color_r |
| 25 | robot | color_g |
| 26 | robot | color_b |
| 27 | robot | z_order |
| 28 | robot | radius |
| 29 | goal_tblock | x |
| 30 | goal_tblock | y |
| 31 | goal_tblock | theta |
| 32 | goal_tblock | vx |
| 33 | goal_tblock | vy |
| 34 | goal_tblock | omega |
| 35 | goal_tblock | static |
| 36 | goal_tblock | held |
| 37 | goal_tblock | color_r |
| 38 | goal_tblock | color_g |
| 39 | goal_tblock | color_b |
| 40 | goal_tblock | z_order |
| 41 | goal_tblock | width |
| 42 | goal_tblock | length_horizontal |
| 43 | goal_tblock | length_vertical |
| 44 | goal_tblock | mass |


## Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Delta x position for robot (positive is right) | -0.050 | 0.050 |
| 1 | dy | Delta y position for robot (positive is up) | -0.050 | 0.050 |


## Rewards
A penalty of -1.0 is given at every time step until the T-block is aligned with the goal pose within specified thresholds.

**Termination Condition**: The episode terminates when all of the following conditions are met:
- Position error in X: |x - x_goal| < 0.03
- Position error in Y: |y - y_goal| < 0.03
- Orientation error: |θ - θ_goal| < 8 degrees

These thresholds ensure the T-block is precisely aligned with the goal pose.


## References
This implementation is based on the Push-T environment introduced in the Diffusion Policy paper (Chi et al., 2023).
