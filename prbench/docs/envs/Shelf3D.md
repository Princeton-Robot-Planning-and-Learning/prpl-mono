# Shelf3D

![random action GIF](assets/random_action_gifs/Shelf3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 3D environment where the goal is to pick up a cube from the ground and place it on a shelf.

## Available Variants
- `prbench/Shelf3D-o1-v0` (o1)
- `prbench/Shelf3D-o2-v0` (o2)
- `prbench/Shelf3D-o3-v0` (o3)
- `prbench/Shelf3D-o5-v0` (o5)
- `prbench/Shelf3D-o10-v0` (o10)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/Shelf3D.gif)

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | pos_base_x |
| 1 | robot | pos_base_y |
| 2 | robot | pos_base_rot |
| 3 | robot | joint_1 |
| 4 | robot | joint_2 |
| 5 | robot | joint_3 |
| 6 | robot | joint_4 |
| 7 | robot | joint_5 |
| 8 | robot | joint_6 |
| 9 | robot | joint_7 |
| 10 | robot | finger_state |
| 11 | robot | grasp_active |
| 12 | robot | grasp_tf_x |
| 13 | robot | grasp_tf_y |
| 14 | robot | grasp_tf_z |
| 15 | robot | grasp_tf_qx |
| 16 | robot | grasp_tf_qy |
| 17 | robot | grasp_tf_qz |
| 18 | robot | grasp_tf_qw |
| 19 | shelf | pose_x |
| 20 | shelf | pose_y |
| 21 | shelf | pose_z |
| 22 | shelf | pose_qx |
| 23 | shelf | pose_qy |
| 24 | shelf | pose_qz |
| 25 | shelf | pose_qw |
| 26 | cube0 | pose_x |
| 27 | cube0 | pose_y |
| 28 | cube0 | pose_z |
| 29 | cube0 | pose_qx |
| 30 | cube0 | pose_qy |
| 31 | cube0 | pose_qz |
| 32 | cube0 | pose_qw |
| 33 | cube0 | grasp_active |
| 34 | cube0 | object_type |
| 35 | cube0 | half_extent_x |
| 36 | cube0 | half_extent_y |
| 37 | cube0 | half_extent_z |
| 38 | cube1 | pose_x |
| 39 | cube1 | pose_y |
| 40 | cube1 | pose_z |
| 41 | cube1 | pose_qx |
| 42 | cube1 | pose_qy |
| 43 | cube1 | pose_qz |
| 44 | cube1 | pose_qw |
| 45 | cube1 | grasp_active |
| 46 | cube1 | object_type |
| 47 | cube1 | half_extent_x |
| 48 | cube1 | half_extent_y |
| 49 | cube1 | half_extent_z |
| 50 | cube2 | pose_x |
| 51 | cube2 | pose_y |
| 52 | cube2 | pose_z |
| 53 | cube2 | pose_qx |
| 54 | cube2 | pose_qy |
| 55 | cube2 | pose_qz |
| 56 | cube2 | pose_qw |
| 57 | cube2 | grasp_active |
| 58 | cube2 | object_type |
| 59 | cube2 | half_extent_x |
| 60 | cube2 | half_extent_y |
| 61 | cube2 | half_extent_z |


## Action Space
An action space for a 7 DOF robot that can open and close its gripper.

    Actions are bounded relative joint positions and open / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.


## Rewards
The reward is a small negative reward (-0.01) per timestep to encourage exploration.

## References
This is a very common kind of environment.
