# TidyBot3D

![random action GIF](assets/random_action_gifs/TidyBot3D.gif)

**Random Action Stats**: Total Reward: -0.25, Success: No, Steps: 25

## Description
A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: table with 5 objects.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Available Variants
This environment has variants that differ in scene type and number of objects. Scene types include 'ground', 'cabinet', etc. The number of objects varies across variants.

- `prbench/TidyBot3D-table-o5-v0` (table-o5)
- `prbench/TidyBot3D-tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_right_side_of_the_kitchen_island-v0` (tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_right_side_of_the_kitchen_island)
- `prbench/TidyBot3D-tool_use-lab2_kitchen-o1-move_the_bowl_to_top_drawer_of_the_kitchen_cabinet-v0` (tool_use-lab2_kitchen-o1-move_the_bowl_to_top_drawer_of_the_kitchen_cabinet)
- `prbench/TidyBot3D-tool_use-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin-v0` (tool_use-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin)
- `prbench/TidyBot3D-tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0` (tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_left_side_of_the_kitchen_island)
- `prbench/TidyBot3D-tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_right_side_of_the_kitchen_counter-v0` (tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_right_side_of_the_kitchen_counter)
- `prbench/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0` (tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island)
- `prbench/TidyBot3D-navigate-namo-o1-v0` (navigate-namo-o1)
- `prbench/TidyBot3D-ground-o1-v0` (ground-o1)
- `prbench/TidyBot3D-cupboard_real-o1-v0` (cupboard_real-o1)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o1-put_the_boxed_drink_on_the_left_side_of_the_bowl-v0` (rearrange-lab2_kitchen-o1-put_the_boxed_drink_on_the_left_side_of_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o1-put_the_boxed_drink_on_the_right_side_of_the_bowl-v0` (rearrange-lab2_kitchen-o1-put_the_boxed_drink_on_the_right_side_of_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o1-put_the_boxed_drink_next_to_the_bowl-v0` (rearrange-lab2_kitchen-o1-put_the_boxed_drink_next_to_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_can_in_front_of_and_the_boxed_drink_behind_the_bowl-v0` (rearrange-lab2_kitchen-o2-put_the_can_in_front_of_and_the_boxed_drink_behind_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o1-put_the_boxed_drink_behind_the_bowl-v0` (rearrange-lab2_kitchen-o1-put_the_boxed_drink_behind_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_boxed_drink_on_the_left_and_the_can_on_the_right_side_of_the_bowl-v0` (rearrange-lab2_kitchen-o2-put_the_boxed_drink_on_the_left_and_the_can_on_the_right_side_of_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0` (rearrange-lab2_kitchen-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_boxed_drink_in_front_of_and_the_can_behind_the_bowl-v0` (rearrange-lab2_kitchen-o2-put_the_boxed_drink_in_front_of_and_the_can_behind_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_can_on_the_left_and_the_boxed_drink_on_the_right_side_of_the_bowl-v0` (rearrange-lab2_kitchen-o2-put_the_can_on_the_left_and_the_boxed_drink_on_the_right_side_of_the_bowl)
- `prbench/TidyBot3D-rearrange-lab2_kitchen-o1-put_the_boxed_drink_in_front_of_the_bowl-v0` (rearrange-lab2_kitchen-o1-put_the_boxed_drink_in_front_of_the_bowl)
- `prbench/TidyBot3D-sort-lab2-o1-fit_the_blocks_in_the_cupboard-v0` (sort-lab2-o1-fit_the_blocks_in_the_cupboard)
- `prbench/TidyBot3D-sort-lab2-o2-fit_the_blocks_in_the_cupboard-v0` (sort-lab2-o2-fit_the_blocks_in_the_cupboard)
- `prbench/TidyBot3D-sort-lab2-o20-sort_the_cluttered_blocks_into_bowls-v0` (sort-lab2-o20-sort_the_cluttered_blocks_into_bowls)
- `prbench/TidyBot3D-sort-lab2-o12-sort_the_blocks_into_the_cupboard-v0` (sort-lab2-o12-sort_the_blocks_into_the_cupboard)
- `prbench/TidyBot3D-sort-lab2-o6-fit_the_blocks_in_the_cupboard-v0` (sort-lab2-o6-fit_the_blocks_in_the_cupboard)
- `prbench/TidyBot3D-ground-o7-v0` (ground-o7)
- `prbench/TidyBot3D-table-o3-v0` (table-o3)
- `prbench/TidyBot3D-cupboard-o8-v0` (cupboard-o8)
- `prbench/TidyBot3D-balance-o4-v0` (balance-o4)
- `prbench/TidyBot3D-ground-o5-v0` (ground-o5)
- `prbench/TidyBot3D-cupboard_real-o4-v0` (cupboard_real-o4)
- `prbench/TidyBot3D-cupboard_real-o8-v0` (cupboard_real-o8)
- `prbench/TidyBot3D-table-o7-v0` (table-o7)
- `prbench/TidyBot3D-ground-o3-v0` (ground-o3)
- `prbench/TidyBot3D-cupboard_real-o2-v0` (cupboard_real-o2)
- `prbench/TidyBot3D-base_motion-o1-v0` (base_motion-o1)

## Initial State Distribution
*(Initial state GIF could not be generated due to rendering issues)*

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
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
| 80 | robot_0 | pos_base_x |
| 81 | robot_0 | pos_base_y |
| 82 | robot_0 | pos_base_rot |
| 83 | robot_0 | pos_arm_joint1 |
| 84 | robot_0 | pos_arm_joint2 |
| 85 | robot_0 | pos_arm_joint3 |
| 86 | robot_0 | pos_arm_joint4 |
| 87 | robot_0 | pos_arm_joint5 |
| 88 | robot_0 | pos_arm_joint6 |
| 89 | robot_0 | pos_arm_joint7 |
| 90 | robot_0 | pos_gripper |
| 91 | robot_0 | vel_base_x |
| 92 | robot_0 | vel_base_y |
| 93 | robot_0 | vel_base_rot |
| 94 | robot_0 | vel_arm_joint1 |
| 95 | robot_0 | vel_arm_joint2 |
| 96 | robot_0 | vel_arm_joint3 |
| 97 | robot_0 | vel_arm_joint4 |
| 98 | robot_0 | vel_arm_joint5 |
| 99 | robot_0 | vel_arm_joint6 |
| 100 | robot_0 | vel_arm_joint7 |
| 101 | robot_0 | vel_gripper |
| 102 | table_1 | x |
| 103 | table_1 | y |
| 104 | table_1 | z |
| 105 | table_1 | qw |
| 106 | table_1 | qx |
| 107 | table_1 | qy |
| 108 | table_1 | qz |
| 109 | table_2 | x |
| 110 | table_2 | y |
| 111 | table_2 | z |
| 112 | table_2 | qw |
| 113 | table_2 | qx |
| 114 | table_2 | qy |
| 115 | table_2 | qz |


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
