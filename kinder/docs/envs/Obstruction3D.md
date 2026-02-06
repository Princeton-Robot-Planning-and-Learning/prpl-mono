# Obstruction3D

![random action GIF](assets/random_action_gifs/Obstruction3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 3D obstruction clearance environment where the goal is to place a target block on a designated target region by first clearing obstructions.

The robot is a Kinova Gen-3 with 7 degrees of freedom that can grasp and manipulate objects. The environment consists of:
- A **table** with dimensions 0.400m × 0.800m × 0.500m
- A **target region** (purple block) with random dimensions between (0.02, 0.02, 0.005) and (0.05, 0.05, 0.005) half-extents
- A **target block** that must be placed on the target region, sized at 0.8× the target region's x,y dimensions
- **Obstruction(s)** (red blocks) that may be placed on or near the target region, blocking access

Obstructions have random dimensions between (0.01, 0.01, 0.01) and (0.02, 0.02, 0.03) half-extents. During initialization, there's a 0.9 probability that each obstruction will be placed on the target region, requiring clearance.

The task requires planning to grasp and move obstructions out of the way, then place the target block on the target region.


## Available Variants
The number of obstructions differs between environment variants. For example, Obstruction3D-o0 has no obstructions, while Obstruction3D-o4 has 4 obstructions.

- [`kinder/Obstruction3D-o0-v0`](variants/Obstruction3D/Obstruction3D-o0.md) (o0)
- [`kinder/Obstruction3D-o1-v0`](variants/Obstruction3D/Obstruction3D-o1.md) (o1)
- [`kinder/Obstruction3D-o2-v0`](variants/Obstruction3D/Obstruction3D-o2.md) (o2)
- [`kinder/Obstruction3D-o3-v0`](variants/Obstruction3D/Obstruction3D-o3.md) (o3)
- [`kinder/Obstruction3D-o4-v0`](variants/Obstruction3D/Obstruction3D-o4.md) (o4)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/Obstruction3D.gif)

## Example Demonstration
![demo GIF](assets/group_gifs/Obstruction3D.gif)

## Observation Space
*(Differs per variant, see individual variant pages)*

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
The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when the target block is placed on the target region (while not being grasped)

The goal is considered reached when:
1. The robot is not currently grasping the target block
2. The target block is resting on (supported by) the target region

Support is determined based on contact between the target block and target region, within a small distance threshold (1e-4).

This encourages the robot to efficiently clear obstructions and place the target block while avoiding infinite episodes.


## References
Similar environments have been used many times, especially in the task and motion planning literature. We took inspiration especially from the "1D Continuous TAMP" environment in [PDDLStream](https://github.com/caelan/pddlstream).
