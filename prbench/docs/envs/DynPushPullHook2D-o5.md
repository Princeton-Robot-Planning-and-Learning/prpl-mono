# prbench/DynPushPullHook2D-o5-v0
![random action GIF](assets/random_action_gifs/DynPushPullHook2D-o5.gif)

### Description
A 2D physics-based tool-use environment where a robot must use a Hook to push/pull a target block onto a middle wall (goal surface). The target block is positioned in the upper region of the world, while the middle wall is located at the center. The robot must manipulate the Hook to navigate the target block downward through obstacles.

The target block is initially surrounded by 5 obstacle blocks that form a barrier around it.

The robot has a movable circular base and an extendable arm with gripper fingers. The Hook is a kinematic object that can be grasped and used as a tool to indirectly manipulate the target block. All dynamic objects follow realistic PyMunk physics including gravity, friction, and collisions.

**Observation Space**: The observation is a fixed-size vector containing the state of all objects:
- **Robot**: position (x,y), orientation (θ), velocities (vx,vy,ω), arm extension, gripper gap
- **Hook**: position, orientation, dimensions (kinematic tool object, can be grasped)
- **Target Block**: position, orientation, velocities, dimensions (dynamic physics object)
- **Middle Wall**: position, orientation, dimensions (kinematic goal surface at world center)
- **Obstruction Blocks** (5): position, orientation, velocities, dimensions (dynamic objects sampled around target)

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/DynPushPullHook2D-o5.gif)

### Example Demonstration
![demo GIF](assets/demo_gifs/DynPushPullHook2D-o5/DynPushPullHook2D-o5_seed0_1759502583.gif)

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | vx_base |
| 4 | robot | vy_base |
| 5 | robot | omega_base |
| 6 | robot | vx_arm |
| 7 | robot | vy_arm |
| 8 | robot | omega_arm |
| 9 | robot | vx_gripper |
| 10 | robot | vy_gripper |
| 11 | robot | omega_gripper |
| 12 | robot | static |
| 13 | robot | base_radius |
| 14 | robot | arm_joint |
| 15 | robot | arm_length |
| 16 | robot | gripper_base_width |
| 17 | robot | gripper_base_height |
| 18 | robot | finger_gap |
| 19 | robot | finger_height |
| 20 | robot | finger_width |
| 21 | hook | x |
| 22 | hook | y |
| 23 | hook | theta |
| 24 | hook | vx |
| 25 | hook | vy |
| 26 | hook | omega |
| 27 | hook | static |
| 28 | hook | held |
| 29 | hook | color_r |
| 30 | hook | color_g |
| 31 | hook | color_b |
| 32 | hook | z_order |
| 33 | hook | width |
| 34 | hook | length_side1 |
| 35 | hook | length_side2 |
| 36 | hook | mass |
| 37 | target_block | x |
| 38 | target_block | y |
| 39 | target_block | theta |
| 40 | target_block | vx |
| 41 | target_block | vy |
| 42 | target_block | omega |
| 43 | target_block | static |
| 44 | target_block | held |
| 45 | target_block | color_r |
| 46 | target_block | color_g |
| 47 | target_block | color_b |
| 48 | target_block | z_order |
| 49 | target_block | width |
| 50 | target_block | height |
| 51 | target_block | mass |
| 52 | obstruction0 | x |
| 53 | obstruction0 | y |
| 54 | obstruction0 | theta |
| 55 | obstruction0 | vx |
| 56 | obstruction0 | vy |
| 57 | obstruction0 | omega |
| 58 | obstruction0 | static |
| 59 | obstruction0 | held |
| 60 | obstruction0 | color_r |
| 61 | obstruction0 | color_g |
| 62 | obstruction0 | color_b |
| 63 | obstruction0 | z_order |
| 64 | obstruction0 | width |
| 65 | obstruction0 | height |
| 66 | obstruction0 | mass |
| 67 | obstruction1 | x |
| 68 | obstruction1 | y |
| 69 | obstruction1 | theta |
| 70 | obstruction1 | vx |
| 71 | obstruction1 | vy |
| 72 | obstruction1 | omega |
| 73 | obstruction1 | static |
| 74 | obstruction1 | held |
| 75 | obstruction1 | color_r |
| 76 | obstruction1 | color_g |
| 77 | obstruction1 | color_b |
| 78 | obstruction1 | z_order |
| 79 | obstruction1 | width |
| 80 | obstruction1 | height |
| 81 | obstruction1 | mass |
| 82 | obstruction2 | x |
| 83 | obstruction2 | y |
| 84 | obstruction2 | theta |
| 85 | obstruction2 | vx |
| 86 | obstruction2 | vy |
| 87 | obstruction2 | omega |
| 88 | obstruction2 | static |
| 89 | obstruction2 | held |
| 90 | obstruction2 | color_r |
| 91 | obstruction2 | color_g |
| 92 | obstruction2 | color_b |
| 93 | obstruction2 | z_order |
| 94 | obstruction2 | width |
| 95 | obstruction2 | height |
| 96 | obstruction2 | mass |
| 97 | obstruction3 | x |
| 98 | obstruction3 | y |
| 99 | obstruction3 | theta |
| 100 | obstruction3 | vx |
| 101 | obstruction3 | vy |
| 102 | obstruction3 | omega |
| 103 | obstruction3 | static |
| 104 | obstruction3 | held |
| 105 | obstruction3 | color_r |
| 106 | obstruction3 | color_g |
| 107 | obstruction3 | color_b |
| 108 | obstruction3 | z_order |
| 109 | obstruction3 | width |
| 110 | obstruction3 | height |
| 111 | obstruction3 | mass |
| 112 | obstruction4 | x |
| 113 | obstruction4 | y |
| 114 | obstruction4 | theta |
| 115 | obstruction4 | vx |
| 116 | obstruction4 | vy |
| 117 | obstruction4 | omega |
| 118 | obstruction4 | static |
| 119 | obstruction4 | held |
| 120 | obstruction4 | color_r |
| 121 | obstruction4 | color_g |
| 122 | obstruction4 | color_b |
| 123 | obstruction4 | z_order |
| 124 | obstruction4 | width |
| 125 | obstruction4 | height |
| 126 | obstruction4 | mass |


### Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.065 | 0.065 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | dgripper | Change in gripper gap (positive is open) | -0.020 | 0.020 |


### Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when the target block reaches the middle wall (goal surface).

**Termination Condition**: The episode terminates when the target block geometrically intersects with the middle wall. This is detected using collision checking between the target block and middle wall.

**Goal Achievement Strategy**: The robot must:
1. Grasp the Hook tool with its gripper
2. Use the Hook to push or pull the target block downward
3. Navigate around or through the obstruction blocks
4. Successfully move the target block until it contacts the middle wall

**Physics Integration**: Since this environment uses PyMunk physics simulation, objects have realistic dynamics including:
- Friction between surfaces
- Collision response and momentum transfer
- Realistic grasping and tool manipulation dynamics
- Indirect manipulation through tool-object interactions
- NOTE: all objects are on a 2D plane with no gravity, but damping is applied to simulate frictional losses


### References
This environment implements a tool-use manipulation task with physics-based dynamics. It is inspired by cognitive science research on tool use and indirect manipulation, where an agent must use an intermediary object (Hook) to achieve goals that cannot be reached directly.

**Key Features**:
- **Tool-Use Paradigm**: Robot must grasp and manipulate a Hook to indirectly move the target block
- **Spatial Reasoning**: Target block starts in upper region, must be moved downward to center goal
- **Obstacle Navigation**: Obstructions are sampled via Gaussian distribution around the target, creating clustered barriers
- **PyMunk Physics Engine**: Provides realistic 2D rigid body dynamics for tool-object interactions
- **Z-Order Collision Control**: Hook and target have surface z-order to avoid collision with floor-level middle wall

**Research Applications**:
- Tool-use learning and reasoning
- Indirect manipulation strategies
- Multi-step planning with intermediate tool grasping
- Physics-aware motion planning through obstacles
- Comparative studies of direct vs. tool-mediated manipulation

This environment enables evaluation of manipulation policies that require tool use, spatial reasoning, and multi-object interaction planning under realistic physics constraints.
