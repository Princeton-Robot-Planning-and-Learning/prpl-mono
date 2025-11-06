# prbench/TidyBot3D-table-o3-v0
![random action GIF](assets/random_action_gifs/TidyBot3D-table-o3.gif)

### Description
A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: table with 3 objects.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/TidyBot3D-table-o3.gif)

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
| 13 | cube2 | x |
| 14 | cube2 | y |
| 15 | cube2 | z |
| 16 | cube2 | qw |
| 17 | cube2 | qx |
| 18 | cube2 | qy |
| 19 | cube2 | qz |
| 20 | cube2 | vx |
| 21 | cube2 | vy |
| 22 | cube2 | vz |
| 23 | cube2 | wx |
| 24 | cube2 | wy |
| 25 | cube2 | wz |
| 26 | cube3 | x |
| 27 | cube3 | y |
| 28 | cube3 | z |
| 29 | cube3 | qw |
| 30 | cube3 | qx |
| 31 | cube3 | qy |
| 32 | cube3 | qz |
| 33 | cube3 | vx |
| 34 | cube3 | vy |
| 35 | cube3 | vz |
| 36 | cube3 | wx |
| 37 | cube3 | wy |
| 38 | cube3 | wz |
| 39 | robot | pos_base_x |
| 40 | robot | pos_base_y |
| 41 | robot | pos_base_rot |
| 42 | robot | pos_arm_joint1 |
| 43 | robot | pos_arm_joint2 |
| 44 | robot | pos_arm_joint3 |
| 45 | robot | pos_arm_joint4 |
| 46 | robot | pos_arm_joint5 |
| 47 | robot | pos_arm_joint6 |
| 48 | robot | pos_arm_joint7 |
| 49 | robot | pos_gripper |
| 50 | robot | vel_base_x |
| 51 | robot | vel_base_y |
| 52 | robot | vel_base_rot |
| 53 | robot | vel_arm_joint1 |
| 54 | robot | vel_arm_joint2 |
| 55 | robot | vel_arm_joint3 |
| 56 | robot | vel_arm_joint4 |
| 57 | robot | vel_arm_joint5 |
| 58 | robot | vel_arm_joint6 |
| 59 | robot | vel_arm_joint7 |
| 60 | robot | vel_gripper |
| 61 | table_1 | x |
| 62 | table_1 | y |
| 63 | table_1 | z |
| 64 | table_1 | qw |
| 65 | table_1 | qx |
| 66 | table_1 | qy |
| 67 | table_1 | qz |
| 68 | table_2 | x |
| 69 | table_2 | y |
| 70 | table_2 | z |
| 71 | table_2 | qw |
| 72 | table_2 | qx |
| 73 | table_2 | qy |
| 74 | table_2 | qz |


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
