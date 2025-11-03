# prbench/TidyBot3D-ground-o7-v0
![random action GIF](assets/random_action_gifs/TidyBot3D-ground-o7.gif)

### Description
A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: ground with 7 objects. In the 'ground' scene, objects are placed randomly on a flat ground plane.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/TidyBot3D-ground-o7.gif)

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
| 16 | cube2 | x |
| 17 | cube2 | y |
| 18 | cube2 | z |
| 19 | cube2 | qw |
| 20 | cube2 | qx |
| 21 | cube2 | qy |
| 22 | cube2 | qz |
| 23 | cube2 | vx |
| 24 | cube2 | vy |
| 25 | cube2 | vz |
| 26 | cube2 | wx |
| 27 | cube2 | wy |
| 28 | cube2 | wz |
| 29 | cube2 | bb_x |
| 30 | cube2 | bb_y |
| 31 | cube2 | bb_z |
| 32 | cube3 | x |
| 33 | cube3 | y |
| 34 | cube3 | z |
| 35 | cube3 | qw |
| 36 | cube3 | qx |
| 37 | cube3 | qy |
| 38 | cube3 | qz |
| 39 | cube3 | vx |
| 40 | cube3 | vy |
| 41 | cube3 | vz |
| 42 | cube3 | wx |
| 43 | cube3 | wy |
| 44 | cube3 | wz |
| 45 | cube3 | bb_x |
| 46 | cube3 | bb_y |
| 47 | cube3 | bb_z |
| 48 | cube4 | x |
| 49 | cube4 | y |
| 50 | cube4 | z |
| 51 | cube4 | qw |
| 52 | cube4 | qx |
| 53 | cube4 | qy |
| 54 | cube4 | qz |
| 55 | cube4 | vx |
| 56 | cube4 | vy |
| 57 | cube4 | vz |
| 58 | cube4 | wx |
| 59 | cube4 | wy |
| 60 | cube4 | wz |
| 61 | cube4 | bb_x |
| 62 | cube4 | bb_y |
| 63 | cube4 | bb_z |
| 64 | cube5 | x |
| 65 | cube5 | y |
| 66 | cube5 | z |
| 67 | cube5 | qw |
| 68 | cube5 | qx |
| 69 | cube5 | qy |
| 70 | cube5 | qz |
| 71 | cube5 | vx |
| 72 | cube5 | vy |
| 73 | cube5 | vz |
| 74 | cube5 | wx |
| 75 | cube5 | wy |
| 76 | cube5 | wz |
| 77 | cube5 | bb_x |
| 78 | cube5 | bb_y |
| 79 | cube5 | bb_z |
| 80 | cube6 | x |
| 81 | cube6 | y |
| 82 | cube6 | z |
| 83 | cube6 | qw |
| 84 | cube6 | qx |
| 85 | cube6 | qy |
| 86 | cube6 | qz |
| 87 | cube6 | vx |
| 88 | cube6 | vy |
| 89 | cube6 | vz |
| 90 | cube6 | wx |
| 91 | cube6 | wy |
| 92 | cube6 | wz |
| 93 | cube6 | bb_x |
| 94 | cube6 | bb_y |
| 95 | cube6 | bb_z |
| 96 | cube7 | x |
| 97 | cube7 | y |
| 98 | cube7 | z |
| 99 | cube7 | qw |
| 100 | cube7 | qx |
| 101 | cube7 | qy |
| 102 | cube7 | qz |
| 103 | cube7 | vx |
| 104 | cube7 | vy |
| 105 | cube7 | vz |
| 106 | cube7 | wx |
| 107 | cube7 | wy |
| 108 | cube7 | wz |
| 109 | cube7 | bb_x |
| 110 | cube7 | bb_y |
| 111 | cube7 | bb_z |
| 112 | robot | pos_base_x |
| 113 | robot | pos_base_y |
| 114 | robot | pos_base_rot |
| 115 | robot | pos_arm_joint1 |
| 116 | robot | pos_arm_joint2 |
| 117 | robot | pos_arm_joint3 |
| 118 | robot | pos_arm_joint4 |
| 119 | robot | pos_arm_joint5 |
| 120 | robot | pos_arm_joint6 |
| 121 | robot | pos_arm_joint7 |
| 122 | robot | pos_gripper |
| 123 | robot | vel_base_x |
| 124 | robot | vel_base_y |
| 125 | robot | vel_base_rot |
| 126 | robot | vel_arm_joint1 |
| 127 | robot | vel_arm_joint2 |
| 128 | robot | vel_arm_joint3 |
| 129 | robot | vel_arm_joint4 |
| 130 | robot | vel_arm_joint5 |
| 131 | robot | vel_arm_joint6 |
| 132 | robot | vel_arm_joint7 |
| 133 | robot | vel_gripper |


### Action Space
Actions: base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)

### Rewards
The primary reward is for successfully placing objects at their target locations.
- A reward of +1.0 is given for each object placed within a 5cm tolerance of its target.
- A smaller positive reward is given for objects within a 10cm tolerance to guide the robot.
- A small negative reward (-0.01) is applied at each timestep to encourage efficiency.
The episode terminates when all objects are placed at their respective targets.


### References
TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
