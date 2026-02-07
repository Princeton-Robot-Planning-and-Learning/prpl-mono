# ScoopPour3D

![random action GIF](assets/random_action_gifs/ScoopPour3D.gif)

**Random Action Stats**: Total Reward: -0.25, Success: No, Steps: 25

## Description
A 3D task where the robot must transfer a pile of objects from one bin to another. There is a tool available that may be used for scooping and pouring.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: lab2_kitchen.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Available Variants
The variants require scooping and pouring different numbers of objects.

- [`kinder/TidyBot3D-ScoopPour3D-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin-v0`](variants/ScoopPour3D/TidyBot3D-ScoopPour3D-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin.md) (TidyBot3D-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/ScoopPour3D.gif)

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
