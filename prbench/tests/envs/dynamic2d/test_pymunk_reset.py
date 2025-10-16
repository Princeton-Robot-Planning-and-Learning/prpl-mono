import pymunk
import numpy as np
from pymunk import Vec2d
from prbench.utils import load_demo

def test_pymunk_simple_env():
    space = pymunk.Space()
    # From PushTee
    space.gravity = 0.0, 0.0
    space.damping = 0.0
    space.collision_slop = 0.001
    control_hz = 10
    sim_hz = 100
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    steps_per_control = sim_hz // control_hz
    kp = 50.0
    kv = 5.0

    # DotRobot
    robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    robot_shape = pymunk.Circle(robot_body, 0.1)
    robot_shape.friction = 1.0
    robot_shape.density = 1.0
    robot_body.position = (0.0, 0.0)
    space.add(robot_body, robot_shape)

    # Randomly Move Robot by 20 times
    actions = [
        np.random.uniform(-0.05, 0.05, size=2) for _ in range(1000)
    ]
    states = [np.array([robot_body.position.x, 
              robot_body.position.y,
              robot_body.velocity.x,
              robot_body.velocity.y], dtype=np.float32)
    ]
    for i in range(len(actions)):
        tgt_pos = np.array([robot_body.position.x,
                            robot_body.position.y], dtype=np.float32) \
                           + actions[i].astype(np.float32)
        tgt_vel = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(steps_per_control):
            curr_pos = np.array([robot_body.position.x,
                            robot_body.position.y], dtype=np.float32)
            curr_vel = np.array([robot_body.velocity.x,
                            robot_body.velocity.y], dtype=np.float32)
            # PD control
            acceleration = kp * (tgt_pos - curr_pos) + kv * (tgt_vel - curr_vel)
            new_vel = curr_vel + acceleration * control_dt
            robot_body.velocity = (
                new_vel[0], new_vel[1]
            )
            for _ in range(sim_hz // control_hz):
                space.step(sim_dt)
        pos_control_err = np.linalg.norm(np.array(tgt_pos) - np.array(robot_body.position))
        print(f"Step {i}, pos err: {pos_control_err}")
        states.append(np.array([robot_body.position.x, 
                        robot_body.position.y,
                        robot_body.velocity.x,
                        robot_body.velocity.y], dtype=np.float32))
    np.savez("state_actions.npz",
        states=np.array(states),
        actions=np.array(actions)
    )

def test_pymunk_simple_env_replay():
    space = pymunk.Space()
    # From PushTee
    space.gravity = 0.0, 0.0
    space.damping = 0.0
    space.collision_slop = 0.001
    control_hz = 10
    sim_hz = 100
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    steps_per_control = sim_hz // control_hz
    kp = 50.0
    kv = 5.0
    data = np.load("state_actions.npz")
    loaded_states = data["states"]
    actions = data["actions"]

    # DotRobot
    robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    robot_shape = pymunk.Circle(robot_body, 0.1)
    robot_shape.friction = 1.0
    robot_shape.density = 1.0
    robot_body.position = (0.0, 0.0)
    space.add(robot_body, robot_shape)

    # MoveRobot forward 5 times
    states = [np.array([robot_body.position.x, 
              robot_body.position.y,
              robot_body.velocity.x,
              robot_body.velocity.y])
    ]
    for i in range(len(actions)):
        tgt_pos = robot_body.position + actions[i]
        tgt_vel = Vec2d(0, 0)
        for _ in range(steps_per_control):
            curr_pos = robot_body.position
            curr_vel = robot_body.velocity
            # PD control
            acceleration = kp * (tgt_pos - curr_pos) + kv * (tgt_vel - curr_vel)
            new_vel = curr_vel + acceleration * control_dt
            robot_body.velocity = new_vel
            for _ in range(sim_hz // control_hz):
                space.step(sim_dt)
        states.append(np.array([robot_body.position.x, 
                        robot_body.position.y,
                        robot_body.velocity.x,
                        robot_body.velocity.y]))
        # error to loaded states
        loaded_state = loaded_states[i+1]
        max_err = np.max(np.abs(loaded_state - states[-1]))
        print(f"Step {i}, max err to loaded state: {max_err}")

def test_pymunk_simple_env_reset():
    space = pymunk.Space()
    # From PushTee
    space.gravity = 0.0, 0.0
    space.damping = 0.0
    space.collision_slop = 0.001
    control_hz = 10
    sim_hz = 100
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    steps_per_control = sim_hz // control_hz
    kp = 50.0
    kv = 5.0
    data = np.load("state_actions.npz")
    loaded_states = data["states"]
    actions = data["actions"]

    # DotRobot
    robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    robot_shape = pymunk.Circle(robot_body, 0.1)
    robot_shape.friction = 1.0
    robot_shape.density = 1.0
    space.add(robot_body, robot_shape)

    # Reset and replay from each intermediate state
    states = []
    for i in range(20):
        loaded_state_prev = loaded_states[i]
        robot_body.position = (loaded_state_prev[0], loaded_state_prev[1])
        robot_body.velocity = (loaded_state_prev[2], loaded_state_prev[3])
        tgt_pos = np.array([robot_body.position.x,
                            robot_body.position.y], dtype=np.float32) \
                           + actions[i].astype(np.float32)
        tgt_vel = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(steps_per_control):
            curr_pos = np.array([robot_body.position.x,
                            robot_body.position.y], dtype=np.float32)
            curr_vel = np.array([robot_body.velocity.x,
                            robot_body.velocity.y], dtype=np.float32)
            # PD control
            acceleration = kp * (tgt_pos - curr_pos) + kv * (tgt_vel - curr_vel)
            new_vel = curr_vel + acceleration * control_dt
            robot_body.velocity = (
                new_vel[0], new_vel[1]
            )
            for _ in range(sim_hz // control_hz):
                space.step(sim_dt)
        states.append(np.array([robot_body.position.x,
                        robot_body.position.y,
                        robot_body.velocity.x,
                        robot_body.velocity.y], dtype=np.float32))
        # error to loaded states
        loaded_state = loaded_states[i+1]
        max_err = np.max(np.abs(loaded_state - states[-1]))
        print(f"Step {i}, max err to loaded state: {max_err}")


def test_pymunk_simple_env_remove_add_reset():
    space = pymunk.Space()
    # From PushTee
    space.gravity = 0.0, 0.0
    space.damping = 0.0
    space.collision_slop = 0.001
    control_hz = 10
    sim_hz = 100
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    steps_per_control = sim_hz // control_hz
    kp = 50.0
    kv = 5.0
    data = np.load("state_actions.npz")
    loaded_states = data["states"]
    actions = data["actions"]

    # DotRobot
    robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    robot_shape = pymunk.Circle(robot_body, 0.1)
    robot_shape.friction = 1.0
    robot_shape.density = 1.0
    space.add(robot_body, robot_shape)

    # Reset and replay from each intermediate state
    states = []
    for i in range(len(actions)):
        loaded_state_prev = loaded_states[i]
        # First remove all the bodies and shapes
        for body in list(space.bodies):
            for shape in list(body.shapes):
                if body in space.bodies:
                    space.remove(body, shape)
        new_space = pymunk.Space()
        new_space.gravity = 0.0, 0.0
        new_space.damping = 0.0
        new_space.collision_slop = 0.001
        robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        robot_shape = pymunk.Circle(robot_body, 0.1)
        robot_shape.friction = 1.0
        robot_shape.density = 1.0
        new_space.add(robot_body, robot_shape)
        robot_body.position = (loaded_state_prev[0], loaded_state_prev[1])
        robot_body.velocity = (loaded_state_prev[2], loaded_state_prev[3])
        tgt_pos = robot_body.position + actions[i]
        tgt_vel = Vec2d(0, 0)
        for _ in range(steps_per_control):
            curr_pos = robot_body.position
            curr_vel = robot_body.velocity
            # PD control
            acceleration = kp * (tgt_pos - curr_pos) + kv * (tgt_vel - curr_vel)
            new_vel = curr_vel + acceleration * control_dt
            robot_body.velocity = new_vel
            for _ in range(sim_hz // control_hz):
                new_space.step(sim_dt)
        states.append(np.array([robot_body.position.x,
                        robot_body.position.y,
                        robot_body.velocity.x,
                        robot_body.velocity.y]))
        # error to loaded states
        loaded_state = loaded_states[i+1]
        max_err = np.max(np.abs(loaded_state - states[-1]))
        print(f"Step {i}, max err to loaded state: {max_err}")


def test_pymunk_simple_env_replay_pushtee():
    # Extract demo information
    demo_path = 'prbench/demos/DynPushT-t1/0/1760636935.p'
    demo_data = load_demo(demo_path)
    actions = demo_data["actions"]
    expected_observations = demo_data["observations"]

    space = pymunk.Space()
    # From PushTee
    space.gravity = 0.0, 0.0
    space.damping = 0.0
    space.collision_slop = 0.001
    control_hz = 10
    sim_hz = 100
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    steps_per_control = sim_hz // control_hz
    kp = 50.0
    kv = 5.0
    loaded_states = []
    for obs in expected_observations:
        loaded_states.append(np.array([obs[16],
                                       obs[17],
                                       obs[19],
                                       obs[20]]))

    # DotRobot
    robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    robot_shape = pymunk.Circle(robot_body, 0.1)
    robot_shape.friction = 1.0
    robot_shape.density = 1.0
    space.add(robot_body, robot_shape)
    robot_body.position = (loaded_states[0][0], loaded_states[0][1])
    robot_body.velocity = (loaded_states[0][2], loaded_states[0][3])

    # Reset and replay from each intermediate state
    states = []
    for i in range(len(actions)):
        tgt_pos = robot_body.position + actions[i]
        tgt_vel = Vec2d(0, 0)
        for _ in range(steps_per_control):
            curr_pos = robot_body.position
            curr_vel = robot_body.velocity
            # PD control
            acceleration = kp * (tgt_pos - curr_pos) + kv * (tgt_vel - curr_vel)
            new_vel = curr_vel + acceleration * control_dt
            robot_body.velocity = new_vel
            for _ in range(sim_hz // control_hz):
                space.step(sim_dt)
        states.append(np.array([robot_body.position.x,
                        robot_body.position.y,
                        robot_body.velocity.x,
                        robot_body.velocity.y]))
        # error to loaded states
        loaded_state = loaded_states[i+1]
        max_err = np.max(np.abs(loaded_state - states[-1]))
        print(f"Step {i}, max err to loaded state: {max_err}")

def test_pymunk_simple_env_reset_pushtee():
    # Extract demo information
    demo_path = 'prbench/demos/DynPushT-t1/0/1760636935.p'
    demo_data = load_demo(demo_path)
    actions = demo_data["actions"]
    expected_observations = demo_data["observations"]

    space = pymunk.Space()
    # From PushTee
    space.gravity = 0.0, 0.0
    space.damping = 0.0
    space.collision_slop = 0.001
    control_hz = 10
    sim_hz = 100
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    steps_per_control = sim_hz // control_hz
    kp = 50.0
    kv = 5.0
    loaded_states = []
    for obs in expected_observations:
        loaded_states.append(np.array([obs[16],
                                       obs[17],
                                       obs[19],
                                       obs[20]]))

    # DotRobot
    robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    robot_shape = pymunk.Circle(robot_body, 0.1)
    robot_shape.friction = 1.0
    robot_shape.density = 1.0
    space.add(robot_body, robot_shape)

    # Reset and replay from each intermediate state
    states = []
    for i in range(len(actions)):
        loaded_state_prev = loaded_states[i]
        robot_body.position = (loaded_state_prev[0], loaded_state_prev[1])
        robot_body.velocity = (loaded_state_prev[2], loaded_state_prev[3])
        tgt_pos = robot_body.position + actions[i]
        tgt_vel = Vec2d(0, 0)
        for _ in range(steps_per_control):
            curr_pos = robot_body.position
            curr_vel = robot_body.velocity
            # PD control
            acceleration = kp * (tgt_pos - curr_pos) + kv * (tgt_vel - curr_vel)
            new_vel = curr_vel + acceleration * control_dt
            robot_body.velocity = new_vel
            for _ in range(sim_hz // control_hz):
                space.step(sim_dt)
        states.append(np.array([robot_body.position.x,
                        robot_body.position.y,
                        robot_body.velocity.x,
                        robot_body.velocity.y]))
        # error to loaded states
        loaded_state = loaded_states[i+1]
        max_err = np.max(np.abs(loaded_state - states[-1]))
        print(f"Step {i}, max err to loaded state: {max_err}")