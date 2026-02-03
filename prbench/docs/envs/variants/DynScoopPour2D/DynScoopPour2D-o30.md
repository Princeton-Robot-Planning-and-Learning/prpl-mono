# DynScoopPour2D-o30

## Usage
```python
import prbench
env = prbench.make("prbench/DynScoopPour2D-o30-v0")
```

## Description
This variant has 30 small objects (15 circles, 15 squares).

## Example Demonstration
![demo GIF](../../assets/demo_gifs/DynScoopPour2D-o30/DynScoopPour2D-o30.gif)

**Demo Stats**: Total Reward: -810.00, Success: No, Steps: 810

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
| 40-249 | small_circle0-14 | (x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, radius, mass) |
| 250-459 | small_square0-14 | (x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, size, mass) |
