# Shelf3D

![random action GIF](assets/random_action_gifs/Shelf3D.gif)

**Random Action Stats**: Total Reward: -0.25, Success: No, Steps: 25

## Description
A 3D task where the robot must pick up objects from the ground and place them onto a space-constrained shelf in a cupboard with three layers.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: cupboard_real with 8 objects.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Available Variants
The variants require picking and placing different numbers of objects.

- [`kinder/TidyBot3D-Shelf3D-cupboard_real-o1-v0`](variants/Shelf3D/TidyBot3D-Shelf3D-cupboard_real-o1.md) (TidyBot3D-cupboard_real-o1)
- [`kinder/TidyBot3D-Shelf3D-cupboard_real-o8-v0`](variants/Shelf3D/TidyBot3D-Shelf3D-cupboard_real-o8.md) (TidyBot3D-cupboard_real-o8)
- [`kinder/TidyBot3D-Shelf3D-cupboard_real-o2-v0`](variants/Shelf3D/TidyBot3D-Shelf3D-cupboard_real-o2.md) (TidyBot3D-cupboard_real-o2)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/Shelf3D.gif)

## Example Demonstration
![demo GIF](assets/group_gifs/Shelf3D.gif)

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
