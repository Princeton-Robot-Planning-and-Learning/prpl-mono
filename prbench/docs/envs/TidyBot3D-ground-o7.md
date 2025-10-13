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
| 7 | cube2 | x |
| 8 | cube2 | y |
| 9 | cube2 | z |
| 10 | cube2 | qw |
| 11 | cube2 | qx |
| 12 | cube2 | qy |
| 13 | cube2 | qz |
| 14 | cube3 | x |
| 15 | cube3 | y |
| 16 | cube3 | z |
| 17 | cube3 | qw |
| 18 | cube3 | qx |
| 19 | cube3 | qy |
| 20 | cube3 | qz |
| 21 | cube4 | x |
| 22 | cube4 | y |
| 23 | cube4 | z |
| 24 | cube4 | qw |
| 25 | cube4 | qx |
| 26 | cube4 | qy |
| 27 | cube4 | qz |
| 28 | cube5 | x |
| 29 | cube5 | y |
| 30 | cube5 | z |
| 31 | cube5 | qw |
| 32 | cube5 | qx |
| 33 | cube5 | qy |
| 34 | cube5 | qz |
| 35 | cube6 | x |
| 36 | cube6 | y |
| 37 | cube6 | z |
| 38 | cube6 | qw |
| 39 | cube6 | qx |
| 40 | cube6 | qy |
| 41 | cube6 | qz |
| 42 | cube7 | x |
| 43 | cube7 | y |
| 44 | cube7 | z |
| 45 | cube7 | qw |
| 46 | cube7 | qx |
| 47 | cube7 | qy |
| 48 | cube7 | qz |


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
