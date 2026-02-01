# DynScoopPour

![random action GIF](assets/random_action_gifs/DynScoopPour.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 2D physics-based tool-use environment where a robot must use an L-shaped hook to scoop small objects from the left side of a middle wall and pour them onto the right side. The middle wall is half the height of the world, allowing objects to be scooped over it.

The robot has a movable circular base and an extendable arm with gripper fingers. The hook is a kinematic object that can be grasped and used as a tool to scoop the small objects. Small objects are dynamic and follow PyMunk physics, but they cannot be grasped directly by the robot.

All objects include physics properties like mass, moment of inertia, and color information for rendering.


## Available Variants
The number of small objects differs between environment variants. For example, DynScoopPour-o10 has 10 small objects, while DynScoopPour-o50 has 50 small objects.

- [`prbench/DynScoopPour-o10-v0`](variants/DynScoopPour/DynScoopPour-o10.md) (o10)
- [`prbench/DynScoopPour-o20-v0`](variants/DynScoopPour/DynScoopPour-o20.md) (o20)
- [`prbench/DynScoopPour-o30-v0`](variants/DynScoopPour/DynScoopPour-o30.md) (o30)
- [`prbench/DynScoopPour-o50-v0`](variants/DynScoopPour/DynScoopPour-o50.md) (o50)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/DynScoopPour.gif)

## Example Demonstration
![demo GIF](assets/demo_gifs/DynScoopPour-o30/DynScoopPour-o30.gif)

**Demo Stats**: Total Reward: -810.00, Success: No, Steps: 810

## Observation Space
*(Differs per variant, see individual variant pages)*

## Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.030 | 0.030 |
| 1 | dy | Change in robot y position (positive is up) | -0.030 | 0.030 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.098 | 0.098 |
| 3 | darm | Change in robot arm length (positive is out) | -0.080 | 0.080 |
| 4 | dgripper | Change in gripper gap (positive is open) | -0.015 | 0.015 |


## Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when at least 50% of the small objects have been moved to the right side of the middle wall.

## References
This is loosely inspired by the Kitchen2D environment from "Active model learning and diverse action sampling for task and motion planning" (Wang et al., 2018).
