# ConstrainedCupboard3D

![random action GIF](assets/random_action_gifs/ConstrainedCupboard3D.gif)

**Random Action Stats**: Total Reward: -0.25, Success: No, Steps: 25

## Description
A 3D task where the robot is supposed to fit multiple long rods into constrained spaces in a cupboard. The cupboard has varying numbers and sizes of rows and columns.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Available Variants
The variants require fitting different number of objects into cupboards of different sizes with varying arrangement of feasible regions at each reset.

- [`kinder/TidyBot3D-ConstrainedCupboard3D-lab2-o1-fit_the_blocks_in_the_cupboard-v0`](variants/ConstrainedCupboard3D/TidyBot3D-ConstrainedCupboard3D-lab2-o1-fit_the_blocks_in_the_cupboard.md) (TidyBot3D-lab2-o1-fit_the_blocks_in_the_cupboard)
- [`kinder/TidyBot3D-ConstrainedCupboard3D-lab2-o2-fit_the_blocks_in_the_cupboard-v0`](variants/ConstrainedCupboard3D/TidyBot3D-ConstrainedCupboard3D-lab2-o2-fit_the_blocks_in_the_cupboard.md) (TidyBot3D-lab2-o2-fit_the_blocks_in_the_cupboard)
- [`kinder/TidyBot3D-ConstrainedCupboard3D-lab2-o6-fit_the_blocks_in_the_cupboard-v0`](variants/ConstrainedCupboard3D/TidyBot3D-ConstrainedCupboard3D-lab2-o6-fit_the_blocks_in_the_cupboard.md) (TidyBot3D-lab2-o6-fit_the_blocks_in_the_cupboard)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/ConstrainedCupboard3D.gif)

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
*(Differs per variant, see individual variant pages)*

## Action Space
Actions: base pos and yaw (3), arm joints (7), gripper pos (1)

## Rewards
Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.


## References
TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
