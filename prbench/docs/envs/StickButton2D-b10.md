# prbench/StickButton2D-b10-v0
![random action GIF](assets/random_action_gifs/StickButton2D-b10.gif)

### Description
A 2D environment where the goal is to touch all buttons, possibly by using a stick for buttons that are out of the robot's direct reach.

In this environment, there are always 10 buttons.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector.

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/StickButton2D-b10.gif)

### Example Demonstration
![demo GIF](assets/demo_gifs/StickButton2D-b10/StickButton2D-b10_seed1_1752262385.gif)

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | base_radius |
| 4 | robot | arm_joint |
| 5 | robot | arm_length |
| 6 | robot | vacuum |
| 7 | robot | gripper_height |
| 8 | robot | gripper_width |
| 9 | stick | x |
| 10 | stick | y |
| 11 | stick | theta |
| 12 | stick | static |
| 13 | stick | color_r |
| 14 | stick | color_g |
| 15 | stick | color_b |
| 16 | stick | z_order |
| 17 | stick | width |
| 18 | stick | height |
| 19 | button0 | x |
| 20 | button0 | y |
| 21 | button0 | theta |
| 22 | button0 | static |
| 23 | button0 | color_r |
| 24 | button0 | color_g |
| 25 | button0 | color_b |
| 26 | button0 | z_order |
| 27 | button0 | radius |
| 28 | button1 | x |
| 29 | button1 | y |
| 30 | button1 | theta |
| 31 | button1 | static |
| 32 | button1 | color_r |
| 33 | button1 | color_g |
| 34 | button1 | color_b |
| 35 | button1 | z_order |
| 36 | button1 | radius |
| 37 | button2 | x |
| 38 | button2 | y |
| 39 | button2 | theta |
| 40 | button2 | static |
| 41 | button2 | color_r |
| 42 | button2 | color_g |
| 43 | button2 | color_b |
| 44 | button2 | z_order |
| 45 | button2 | radius |
| 46 | button3 | x |
| 47 | button3 | y |
| 48 | button3 | theta |
| 49 | button3 | static |
| 50 | button3 | color_r |
| 51 | button3 | color_g |
| 52 | button3 | color_b |
| 53 | button3 | z_order |
| 54 | button3 | radius |
| 55 | button4 | x |
| 56 | button4 | y |
| 57 | button4 | theta |
| 58 | button4 | static |
| 59 | button4 | color_r |
| 60 | button4 | color_g |
| 61 | button4 | color_b |
| 62 | button4 | z_order |
| 63 | button4 | radius |
| 64 | button5 | x |
| 65 | button5 | y |
| 66 | button5 | theta |
| 67 | button5 | static |
| 68 | button5 | color_r |
| 69 | button5 | color_g |
| 70 | button5 | color_b |
| 71 | button5 | z_order |
| 72 | button5 | radius |
| 73 | button6 | x |
| 74 | button6 | y |
| 75 | button6 | theta |
| 76 | button6 | static |
| 77 | button6 | color_r |
| 78 | button6 | color_g |
| 79 | button6 | color_b |
| 80 | button6 | z_order |
| 81 | button6 | radius |
| 82 | button7 | x |
| 83 | button7 | y |
| 84 | button7 | theta |
| 85 | button7 | static |
| 86 | button7 | color_r |
| 87 | button7 | color_g |
| 88 | button7 | color_b |
| 89 | button7 | z_order |
| 90 | button7 | radius |
| 91 | button8 | x |
| 92 | button8 | y |
| 93 | button8 | theta |
| 94 | button8 | static |
| 95 | button8 | color_r |
| 96 | button8 | color_g |
| 97 | button8 | color_b |
| 98 | button8 | z_order |
| 99 | button8 | radius |
| 100 | button9 | x |
| 101 | button9 | y |
| 102 | button9 | theta |
| 103 | button9 | static |
| 104 | button9 | color_r |
| 105 | button9 | color_g |
| 106 | button9 | color_b |
| 107 | button9 | z_order |
| 108 | button9 | radius |


### Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | vac | Directly sets the vacuum (0.0 is off, 1.0 is on) | 0.000 | 1.000 |


### Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when all buttons have been pressed.


### References
This environment is based on the Stick Button environment that was originally introduced in "Learning Neuro-Symbolic Skills for Bilevel Planning" (Silver et al., CoRL 2022). This version is simplified in that the robot or stick need only make contact with a button to press it, rather than explicitly pressing. Also, the full stick works for pressing, not just the tip.
