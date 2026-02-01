# TidyBot3D-dynamic-lab2-o1-toss_the_blocks_into_the_bin

## Usage
```python
import prbench
env = prbench.make("prbench/TidyBot3D-dynamic-lab2-o1-toss_the_blocks_into_the_bin-v0")
```

## Description
No variant-specific description available.

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | bin_0 | x |
| 1 | bin_0 | y |
| 2 | bin_0 | z |
| 3 | bin_0 | qw |
| 4 | bin_0 | qx |
| 5 | bin_0 | qy |
| 6 | bin_0 | qz |
| 7 | bin_0 | vx |
| 8 | bin_0 | vy |
| 9 | bin_0 | vz |
| 10 | bin_0 | wx |
| 11 | bin_0 | wy |
| 12 | bin_0 | wz |
| 13 | bin_0 | bb_x |
| 14 | bin_0 | bb_y |
| 15 | bin_0 | bb_z |
| 16 | cube_0 | x |
| 17 | cube_0 | y |
| 18 | cube_0 | z |
| 19 | cube_0 | qw |
| 20 | cube_0 | qx |
| 21 | cube_0 | qy |
| 22 | cube_0 | qz |
| 23 | cube_0 | vx |
| 24 | cube_0 | vy |
| 25 | cube_0 | vz |
| 26 | cube_0 | wx |
| 27 | cube_0 | wy |
| 28 | cube_0 | wz |
| 29 | cube_0 | bb_x |
| 30 | cube_0 | bb_y |
| 31 | cube_0 | bb_z |
| 32 | robot | pos_base_x |
| 33 | robot | pos_base_y |
| 34 | robot | pos_base_rot |
| 35 | robot | pos_arm_joint1 |
| 36 | robot | pos_arm_joint2 |
| 37 | robot | pos_arm_joint3 |
| 38 | robot | pos_arm_joint4 |
| 39 | robot | pos_arm_joint5 |
| 40 | robot | pos_arm_joint6 |
| 41 | robot | pos_arm_joint7 |
| 42 | robot | pos_gripper |
| 43 | robot | vel_base_x |
| 44 | robot | vel_base_y |
| 45 | robot | vel_base_rot |
| 46 | robot | vel_arm_joint1 |
| 47 | robot | vel_arm_joint2 |
| 48 | robot | vel_arm_joint3 |
| 49 | robot | vel_arm_joint4 |
| 50 | robot | vel_arm_joint5 |
| 51 | robot | vel_arm_joint6 |
| 52 | robot | vel_arm_joint7 |
| 53 | robot | vel_gripper |
