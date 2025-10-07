# prbench/DynPushT-t1-v0
![random action GIF](assets/random_action_gifs/DynPushT-t1.gif)

### Description
A 2D physics-based environment where the goal is to push a T-shaped block to match a goal pose using a simple dot robot (kinematic circle) with PyMunk physics simulation.

**Observation Space**: The observation is a fixed-size vector containing the state of all objects:
- **Robot**: position (x,y), velocities (vx,vy)
- **T-Block**: position (x,y), orientation (θ), velocities (vx,vy,ω), dimensions (width, length_horizontal, length_vertical) (dynamic physics object)

Each object includes physics properties like mass, moment of inertia, and color information for rendering.

**Task**: Push the T-shaped block so that it covers at least 95% of the goal pose (position and orientation).

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/DynPushT-t1.gif)

### Example Demonstration
*(No demonstration GIFs available)*

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | tblock | x |
| 1 | tblock | y |
| 2 | tblock | theta |
| 3 | tblock | vx |
| 4 | tblock | vy |
| 5 | tblock | omega |
| 6 | tblock | static |
| 7 | tblock | held |
| 8 | tblock | color_r |
| 9 | tblock | color_g |
| 10 | tblock | color_b |
| 11 | tblock | z_order |
| 12 | tblock | width |
| 13 | tblock | length_horizontal |
| 14 | tblock | length_vertical |
| 15 | tblock | mass |
| 16 | robot | x |
| 17 | robot | y |
| 18 | robot | theta |
| 19 | robot | vx |
| 20 | robot | vy |
| 21 | robot | omega |
| 22 | robot | static |
| 23 | robot | held |
| 24 | robot | color_r |
| 25 | robot | color_g |
| 26 | robot | color_b |
| 27 | robot | z_order |
| 28 | robot | radius |
| 29 | goal_tblock | x |
| 30 | goal_tblock | y |
| 31 | goal_tblock | theta |
| 32 | goal_tblock | vx |
| 33 | goal_tblock | vy |
| 34 | goal_tblock | omega |
| 35 | goal_tblock | static |
| 36 | goal_tblock | held |
| 37 | goal_tblock | color_r |
| 38 | goal_tblock | color_g |
| 39 | goal_tblock | color_b |
| 40 | goal_tblock | z_order |
| 41 | goal_tblock | width |
| 42 | goal_tblock | length_horizontal |
| 43 | goal_tblock | length_vertical |
| 44 | goal_tblock | mass |


### Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Delta x position for robot (positive is right) | -0.050 | 0.050 |
| 1 | dy | Delta y position for robot (positive is up) | -0.050 | 0.050 |


### Rewards
The reward is based on the coverage of the T-block with respect to the goal pose, computed using Shapely geometric intersection.

**Reward Formula**: reward = clip(coverage / success_threshold, 0, 1)

where coverage = intersection_area / goal_area

**Termination Condition**: The episode terminates when the T-block achieves at least 95% coverage of the goal pose.

**Physics Integration**: Since this environment uses PyMunk physics simulation, objects have realistic dynamics including:
- No gravity (planar pushing task)
- Friction between surfaces
- Collision response and momentum transfer
- Realistic pushing dynamics


### References
This is a physics-based version of the PushT environment, commonly used in robot manipulation and imitation learning research.

**Original PushT Environment**:
- Introduced in the Diffusion Policy paper
- Features a simple kinematic agent pushing a T-shaped block
- Goal is to match a target pose with 95% coverage
- Uses PyMunk 2D physics engine

**Key Features**:
- **PyMunk Physics Engine**: Provides realistic 2D rigid body dynamics
- **Dynamic T-shaped Block**: Has mass, inertia, and responds to forces
- **Kinematic Dot Robot**: Simple circle that moves to target positions via PD control
- **Coverage-based Reward**: Uses geometric intersection for precise goal checking
- **No Gravity**: Planar manipulation task without gravitational effects

**Research Applications**:
- Robot manipulation learning
- Imitation learning and behavior cloning
- Diffusion policy training and evaluation
- Planar pushing strategy development
- Physics-based motion planning validation

This environment is widely used for evaluating manipulation policies, particularly in the context of diffusion-based and transformer-based imitation learning approaches.
