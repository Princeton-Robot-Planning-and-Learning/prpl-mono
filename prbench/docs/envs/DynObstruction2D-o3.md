# prbench/DynObstruction2D-o3-v0
![random action GIF](assets/random_action_gifs/DynObstruction2D-o3.gif)

### Description
A 2D physics-based environment where the goal is to place a target block onto a target surface using a fingered robot with PyMunk physics simulation. The block must be completely on the surface.

The target surface may be initially obstructed. In this environment, there are always 3 obstacle blocks.

The robot has a movable circular base and an extendable arm with gripper fingers. Objects can be grasped and released through gripper actions. All objects follow realistic physics including gravity, friction, and collisions.

**Observation Space**: The observation is a fixed-size vector containing the state of all objects:
- **Robot**: position (x,y), orientation (θ), velocities (vx,vy,ω), arm extension, gripper gap
- **Target Block**: position, orientation, velocities, dimensions (dynamic physics object)
- **Target Surface**: position, orientation, dimensions (kinematic physics object)
- **Obstruction Blocks** (3): position, orientation, velocities, dimensions (dynamic physics objects)

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/DynObstruction2D-o3.gif)

### Example Demonstration
![demo GIF](assets/demo_gifs/DynObstruction2D-o3/DynObstruction2D-o3_seed7_1757345386.gif)

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | target_surface | x |
| 1 | target_surface | y |
| 2 | target_surface | theta |
| 3 | target_surface | vx |
| 4 | target_surface | vy |
| 5 | target_surface | omega |
| 6 | target_surface | static |
| 7 | target_surface | color_r |
| 8 | target_surface | color_g |
| 9 | target_surface | color_b |
| 10 | target_surface | z_order |
| 11 | target_surface | width |
| 12 | target_surface | height |
| 13 | target_block | x |
| 14 | target_block | y |
| 15 | target_block | theta |
| 16 | target_block | vx |
| 17 | target_block | vy |
| 18 | target_block | omega |
| 19 | target_block | static |
| 20 | target_block | color_r |
| 21 | target_block | color_g |
| 22 | target_block | color_b |
| 23 | target_block | z_order |
| 24 | target_block | width |
| 25 | target_block | height |
| 26 | target_block | mass |
| 27 | obstruction0 | x |
| 28 | obstruction0 | y |
| 29 | obstruction0 | theta |
| 30 | obstruction0 | vx |
| 31 | obstruction0 | vy |
| 32 | obstruction0 | omega |
| 33 | obstruction0 | static |
| 34 | obstruction0 | color_r |
| 35 | obstruction0 | color_g |
| 36 | obstruction0 | color_b |
| 37 | obstruction0 | z_order |
| 38 | obstruction0 | width |
| 39 | obstruction0 | height |
| 40 | obstruction0 | mass |
| 41 | obstruction1 | x |
| 42 | obstruction1 | y |
| 43 | obstruction1 | theta |
| 44 | obstruction1 | vx |
| 45 | obstruction1 | vy |
| 46 | obstruction1 | omega |
| 47 | obstruction1 | static |
| 48 | obstruction1 | color_r |
| 49 | obstruction1 | color_g |
| 50 | obstruction1 | color_b |
| 51 | obstruction1 | z_order |
| 52 | obstruction1 | width |
| 53 | obstruction1 | height |
| 54 | obstruction1 | mass |
| 55 | obstruction2 | x |
| 56 | obstruction2 | y |
| 57 | obstruction2 | theta |
| 58 | obstruction2 | vx |
| 59 | obstruction2 | vy |
| 60 | obstruction2 | omega |
| 61 | obstruction2 | static |
| 62 | obstruction2 | color_r |
| 63 | obstruction2 | color_g |
| 64 | obstruction2 | color_b |
| 65 | obstruction2 | z_order |
| 66 | obstruction2 | width |
| 67 | obstruction2 | height |
| 68 | obstruction2 | mass |
| 69 | robot | x |
| 70 | robot | y |
| 71 | robot | theta |
| 72 | robot | vx |
| 73 | robot | vy |
| 74 | robot | omega |
| 75 | robot | static |
| 76 | robot | base_radius |
| 77 | robot | arm_joint |
| 78 | robot | arm_length |
| 79 | robot | gripper_base_width |
| 80 | robot | gripper_base_height |
| 81 | robot | finger_gap |
| 82 | robot | finger_height |
| 83 | robot | finger_width |


### Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | dgripper | Change in gripper gap (positive is open) | -0.020 | 0.020 |


### Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when the target block is completely "on" the target surface.

**Termination Condition**: The episode terminates when the target block is successfully placed on the target surface. The "on" condition requires that the bottom vertices of the target block are within the bounds of the target surface, accounting for physics-based positioning.

The definition of "on" is implemented using geometric collision detection:
```python
def is_on(
    state: ObjectCentricState,
    top: Object,
    bottom: Object,
    static_object_cache: dict[Object, MultiBody2D],
    tol: float = 0.025,
) -> bool:
    """Checks top object is completely on the bottom one.

    Only rectangles are currently supported.

    Assumes that "up" is positive y.
    """
    top_geom = rectangle_object_to_geom(state, top, static_object_cache)
    bottom_geom = rectangle_object_to_geom(state, bottom, static_object_cache)
    # The bottom-most vertices of top_geom should be contained within the bottom
    # geom when those vertices are offset by tol.
    sorted_vertices = sorted(top_geom.vertices, key=lambda v: v[1])
    for x, y in sorted_vertices[:2]:
        offset_y = y - tol
        if not bottom_geom.contains_point(x, offset_y):
            return False
    return True
```

**Physics Integration**: Since this environment uses PyMunk physics simulation, objects have realistic dynamics including:
- Gravity (objects fall if not supported)
- Friction between surfaces
- Collision response and momentum transfer
- Realistic grasping and manipulation dynamics


### References
This is a physics-based version of manipulation environments commonly used in robotics research. It extends the geometric obstruction environment to include realistic physics simulation using PyMunk.

**Key Features**:
- **PyMunk Physics Engine**: Provides realistic 2D rigid body dynamics
- **Dynamic Objects**: Target and obstruction blocks have mass, inertia, and respond to forces
- **Kinematic Robot**: Multi-DOF robot with base movement, arm extension, and gripper control
- **Collision Detection**: Physics-based collision handling for grasping and object interactions
- **Gravity Simulation**: Objects fall and settle naturally under gravitational forces

**Research Applications**:
- Robot manipulation learning with realistic physics
- Grasping and placement strategy development  
- Multi-object interaction scenarios
- Physics-aware motion planning validation
- Comparative studies between geometric and physics-based environments

This environment enables more realistic evaluation of manipulation policies compared to purely geometric versions, as agents must account for momentum, friction, and gravitational effects.
