# prbench/TidyBot3D-cupboard-o1-real-v0
![random action GIF](assets/random_action_gifs/TidyBot3D-cupboard-o1-real.gif)

### Description
A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: cupboard with 1 objects.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/TidyBot3D-cupboard-o1-real.gif)

### Example Demonstration
*(No demonstration GIFs available)*

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | cube1 | x |
| 1 | cube1 | y |
| 2 | cube1 | z |
| 3 | cube1 | qw |
| 4 | cube1 | qx |
| 5 | cube1 | qy |
| 6 | cube1 | qz |
| 7 | cube1 | vx |
| 8 | cube1 | vy |
| 9 | cube1 | vz |
| 10 | cube1 | wx |
| 11 | cube1 | wy |
| 12 | cube1 | wz |
| 13 | cube1 | bb_x |
| 14 | cube1 | bb_y |
| 15 | cube1 | bb_z |
| 16 | cupboard_1 | x |
| 17 | cupboard_1 | y |
| 18 | cupboard_1 | z |
| 19 | cupboard_1 | qw |
| 20 | cupboard_1 | qx |
| 21 | cupboard_1 | qy |
| 22 | cupboard_1 | qz |
| 23 | robot | pos_base_x |
| 24 | robot | pos_base_y |
| 25 | robot | pos_base_rot |
| 26 | robot | pos_arm_joint1 |
| 27 | robot | pos_arm_joint2 |
| 28 | robot | pos_arm_joint3 |
| 29 | robot | pos_arm_joint4 |
| 30 | robot | pos_arm_joint5 |
| 31 | robot | pos_arm_joint6 |
| 32 | robot | pos_arm_joint7 |
| 33 | robot | pos_gripper |
| 34 | robot | vel_base_x |
| 35 | robot | vel_base_y |
| 36 | robot | vel_base_rot |
| 37 | robot | vel_arm_joint1 |
| 38 | robot | vel_arm_joint2 |
| 39 | robot | vel_arm_joint3 |
| 40 | robot | vel_arm_joint4 |
| 41 | robot | vel_arm_joint5 |
| 42 | robot | vel_arm_joint6 |
| 43 | robot | vel_arm_joint7 |
| 44 | robot | vel_gripper |


### Action Space
Actions: base pos and yaw (3), arm joints (7), gripper pos (1)

### Rewards
Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.


### References
TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
