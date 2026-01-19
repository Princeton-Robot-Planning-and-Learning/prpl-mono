# BaseMotion3D

![random action GIF](assets/random_action_gifs/BaseMotion3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A very simple environment where only base motion planning is needed to reach a goal.

## Available Variants
This environment has only one variant.

- `prbench/BaseMotion3D-v0` (v0)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/BaseMotion3D.gif)

## Example Demonstration
![demo GIF](assets/demo_gifs/BaseMotion3D/BaseMotion3D_1768756176.gif)

**Demo Stats**: Total Reward: -10.00, Success: Yes, Steps: 10

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
| 19 | target | x |
| 20 | target | y |
| 21 | target | z |


## Action Space
An action space for mobile manipulation with a 7 DOF robot that can open and close its gripper.

Actions are bounded relative base position, rotation, and joint positions, and open / close.

| **Index** | **Description** |
| --- | --- |
| 0 | delta base x |
| 1 | delta base y |
| 2 | delta base rotation |
| 3 | delta joint 1 |
| 4 | delta joint 2 |
| 5 | delta joint 3 |
| 6 | delta joint 4 |
| 7 | delta joint 5 |
| 8 | delta joint 6 |
| 9 | delta joint 7 |
| 10 | gripper open/close |

The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.


## Rewards
The reward is -1 per timestep to encourage reaching the goal quickly.

## References
This is a very common kind of environment. The background is adapted from the [Replica dataset](https://arxiv.org/abs/1906.05797) (Straub et al., 2019).
