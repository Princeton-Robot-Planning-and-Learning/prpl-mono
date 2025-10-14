# prbench/TidyBot3D-cupboard-o8-v0
![random action GIF](assets/random_action_gifs/TidyBot3D-cupboard-o8.gif)

### Description
A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: cupboard with 8 objects.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/TidyBot3D-cupboard-o8.gif)

### Example Demonstration
*(No demonstration GIFs available)*

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | pos_base_x |
| 1 | robot | pos_base_y |
| 2 | robot | pos_base_rot |
| 3 | robot | pos_arm_joint1 |
| 4 | robot | pos_arm_joint2 |
| 5 | robot | pos_arm_joint3 |
| 6 | robot | pos_arm_joint4 |
| 7 | robot | pos_arm_joint5 |
| 8 | robot | pos_arm_joint6 |
| 9 | robot | pos_arm_joint7 |
| 10 | robot | pos_gripper |
| 11 | robot | vel_base_x |
| 12 | robot | vel_base_y |
| 13 | robot | vel_base_rot |
| 14 | robot | vel_arm_joint1 |
| 15 | robot | vel_arm_joint2 |
| 16 | robot | vel_arm_joint3 |
| 17 | robot | vel_arm_joint4 |
| 18 | robot | vel_arm_joint5 |
| 19 | robot | vel_arm_joint6 |
| 20 | robot | vel_arm_joint7 |
| 21 | robot | vel_gripper |


### Action Space
Actions: base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)

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
