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
| 39 | cube4 | x |
| 40 | cube4 | y |
| 41 | cube4 | z |
| 42 | cube4 | qw |
| 43 | cube4 | qx |
| 44 | cube4 | qy |
| 45 | cube4 | qz |
| 46 | cube4 | vx |
| 47 | cube4 | vy |
| 48 | cube4 | vz |
| 49 | cube4 | wx |
| 50 | cube4 | wy |
| 51 | cube4 | wz |
| 52 | cube5 | x |
| 53 | cube5 | y |
| 54 | cube5 | z |
| 55 | cube5 | qw |
| 56 | cube5 | qx |
| 57 | cube5 | qy |
| 58 | cube5 | qz |
| 59 | cube5 | vx |
| 60 | cube5 | vy |
| 61 | cube5 | vz |
| 62 | cube5 | wx |
| 63 | cube5 | wy |
| 64 | cube5 | wz |
| 65 | cube6 | x |
| 66 | cube6 | y |
| 67 | cube6 | z |
| 68 | cube6 | qw |
| 69 | cube6 | qx |
| 70 | cube6 | qy |
| 71 | cube6 | qz |
| 72 | cube6 | vx |
| 73 | cube6 | vy |
| 74 | cube6 | vz |
| 75 | cube6 | wx |
| 76 | cube6 | wy |
| 77 | cube6 | wz |
| 78 | cube7 | x |
| 79 | cube7 | y |
| 80 | cube7 | z |
| 81 | cube7 | qw |
| 82 | cube7 | qx |
| 83 | cube7 | qy |
| 84 | cube7 | qz |
| 85 | cube7 | vx |
| 86 | cube7 | vy |
| 87 | cube7 | vz |
| 88 | cube7 | wx |
| 89 | cube7 | wy |
| 90 | cube7 | wz |
| 91 | cube8 | x |
| 92 | cube8 | y |
| 93 | cube8 | z |
| 94 | cube8 | qw |
| 95 | cube8 | qx |
| 96 | cube8 | qy |
| 97 | cube8 | qz |
| 98 | cube8 | vx |
| 99 | cube8 | vy |
| 100 | cube8 | vz |
| 101 | cube8 | wx |
| 102 | cube8 | wy |
| 103 | cube8 | wz |
| 104 | robot | pos_base_x |
| 105 | robot | pos_base_y |
| 106 | robot | pos_base_rot |
| 107 | robot | pos_arm_joint1 |
| 108 | robot | pos_arm_joint2 |
| 109 | robot | pos_arm_joint3 |
| 110 | robot | pos_arm_joint4 |
| 111 | robot | pos_arm_joint5 |
| 112 | robot | pos_arm_joint6 |
| 113 | robot | pos_arm_joint7 |
| 114 | robot | pos_gripper |
| 115 | robot | vel_base_x |
| 116 | robot | vel_base_y |
| 117 | robot | vel_base_rot |
| 118 | robot | vel_arm_joint1 |
| 119 | robot | vel_arm_joint2 |
| 120 | robot | vel_arm_joint3 |
| 121 | robot | vel_arm_joint4 |
| 122 | robot | vel_arm_joint5 |
| 123 | robot | vel_arm_joint6 |
| 124 | robot | vel_arm_joint7 |
| 125 | robot | vel_gripper |


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
