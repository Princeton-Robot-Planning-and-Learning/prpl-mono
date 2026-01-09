# Packing3D

![random action GIF](assets/random_action_gifs/Packing3D.gif)

## Description
A 3D packing environment where the goal is to place a set of parts into a rack without collisions.

The robot is a Kinova Gen-3 with 7 degrees of freedom that can grasp and manipulate objects. The environment consists of:
- A **table** with dimensions 0.400m × 0.800m × 0.500m
- A **rack** (purple) with half-extents (0.1, 0.15, 0.02)
- **Parts** (green) that must be packed into the rack. Parts are sampled with half-extents in (0.05, 0.05, 0.01, 0) to (0.05, 0.05, 0.01, 0) and a probability 0.5 of being triangle-shaped (triangles are represented as triangular prisms with depth 0.020m when used).

The task requires planning to grasp and place each part into the rack while avoiding collisions and ensuring parts are supported by the rack (on the rack and not grasped) at the end.


## Available Variants
- `prbench/Packing3D-p1-v0` (p1)
- `prbench/Packing3D-p2-v0` (p2)
- `prbench/Packing3D-p3-v0` (p3)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/Packing3D.gif)

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
| 19 | rack | pose_x |
| 20 | rack | pose_y |
| 21 | rack | pose_z |
| 22 | rack | pose_qx |
| 23 | rack | pose_qy |
| 24 | rack | pose_qz |
| 25 | rack | pose_qw |
| 26 | rack | grasp_active |
| 27 | rack | object_type |
| 28 | rack | half_extent_x |
| 29 | rack | half_extent_y |
| 30 | rack | half_extent_z |
| 31 | part0 | pose_x |
| 32 | part0 | pose_y |
| 33 | part0 | pose_z |
| 34 | part0 | pose_qx |
| 35 | part0 | pose_qy |
| 36 | part0 | pose_qz |
| 37 | part0 | pose_qw |
| 38 | part0 | grasp_active |
| 39 | part0 | object_type |
| 40 | part0 | half_extent_x |
| 41 | part0 | half_extent_y |
| 42 | part0 | half_extent_z |
| 43 | part1 | pose_x |
| 44 | part1 | pose_y |
| 45 | part1 | pose_z |
| 46 | part1 | pose_qx |
| 47 | part1 | pose_qy |
| 48 | part1 | pose_qz |
| 49 | part1 | pose_qw |
| 50 | part1 | grasp_active |
| 51 | part1 | triangle_type |
| 52 | part1 | side_a |
| 53 | part1 | side_b |
| 54 | part1 | depth |


## Action Space
An action space for a 7 DOF robot that can open and close its gripper.

    Actions are bounded relative joint positions and open / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.


## Rewards
The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when all parts are placed in the rack and none are grasped

The goal is considered reached when:
1. The robot is not currently grasping any part
2. Every part is resting on (supported by) the rack surface

Support is determined based on contact between a part and the rack within a small distance threshold (configured by the environment).

This encourages the robot to efficiently pack the parts into the rack while avoiding infinite episodes.


## References
Packing tasks are common in robotics and automated warehousing literature. This environment is inspired by standard manipulation benchmarks and simple bin-packing problems; it’s intended as a deterministic, physics-based testbed for pick-and-place planning and task-and-motion planning approaches.
