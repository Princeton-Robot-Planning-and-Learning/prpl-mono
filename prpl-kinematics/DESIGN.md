# prpl-kinematics — design

A ground-up successor to `pybullet-helpers`. Kinematics only, engine-agnostic,
built around one general scene graph. This document is the plan; code is added
one milestone at a time, and **nothing lands without unit tests**.

## Principles

1. **Kinematic only.** No dynamics, no `stepSimulation`. State is set by reset.
2. **One scene graph; everything is an edge.** A robot joint, a mobile base, and
   a grasp are all just edges in a `KinematicTree`.
3. **spatialmath is the substrate.** Poses are `SE3`/`SE2`, rotations `SO3`/
   `UnitQuaternion`. We do not define our own Pose or Quaternion classes.
4. **PyBullet is a backend, not the source of truth.** The tree owns forward
   kinematics; PyBullet (and, later, others) are pluggable collision/render
   backends behind an interface.
5. **Interfaces are subclassable.** IK, motion planning, and manipulation
   primitives are ABCs with swappable implementations (e.g. BiRRT vs OMPL).
6. **Start minimal, add as we go.** Every class and method must be necessary
   *now* and covered by a unit test. Speculative abstractions live in this doc,
   not in code.
7. **Configurations are name-keyed; joint groups are explicit.** A
   `Configuration` is a `Mapping[str, JointValues]`, never a bare positional
   vector whose length silently implies which joints it covers. "Does this
   include the gripper?" is answered by which names are keys, not by counting
   values. Flat vectors exist only *inside* a `JointSpace`, which always carries
   its ordered name list. The gripper is just another named joint group, never a
   special `finger_state` scalar — a lesson learned from pybullet-helpers, where
   an ambiguous "robot joints" vector (7 vs 13) plus a bolted-on finger state
   caused constant confusion.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Package name | `prpl_kinematics`, in the `prpl-mono` monorepo |
| Transform math | `spatialmath-python` |
| PyBullet's role | Collision/render **backend** behind an interface |
| Engine of truth | The `KinematicTree` (FK via spatialmath) |
| Proving-set robots | Panda, Kinova, Dexmate Vega (bimanual), TidyBot (mobile) |

## The KinematicTree (the centerpiece)

A directed tree of named frames (`Node`) rooted at `world`. Each non-root node
has exactly one incoming `Edge` carrying a `Joint`. The union of all joints with
`num_dof > 0` is the configuration. One structure unifies cases other libraries
keep separate:

```
world (root)
 ├─[planar joint]→ base ─[fixed]→ torso ─[revolute×7]→ … → gripper_link
 │                                                            └─[fixed: grasp]→ mug   ← re-parented on grasp
 ├─[fixed]→ table
 └─[fixed]→ block
```

- **Robot arm** — a chain of revolute/prismatic edges.
- **Mobile base** — a `PlanarJoint` (SE(2)) edge from `world` to the base frame.
- **Grasp** — `attach()`: re-parent an object's edge onto a gripper frame with a
  `FixedJoint` holding the grasp transform. Release is `attach()` back to `world`.
- **Bimanual / multi-robot / scene** — one tree (or forest under `world`); the
  backend never sees morphology.

Forward kinematics composes joint transforms along the path from the root. The
tree owns no physics.

### Why PyBullet-as-backend is clean here

Because the tree provides the world-frame pose of every geometry-bearing node,
the collision backend never needs joints or articulation: it holds **one
collision shape per node** and, per query, positions each shape from the tree's
FK before running collision detection. Consequences:

- Grasps need no special handling — a held object's shape just follows FK.
- Mobile / bimanual / multi-robot all use one code path.
- The backend is swappable (PyBullet today; FCL/meshcat later) and can run
  headless without a simulator process.

## Layered structure (target)

```
geometry/        spatialmath conversions (PyBullet xyzw) + shape specs.          [built]
tree/            KinematicTree, Joint types, FK, attach (grasp), KinematicState.  [built]
loading/         URDF -> KinematicTree with per-node geometry (via yourdfpy).     [built]
meshes.py        Prepare meshes for PyBullet's file importer (convert/cache).      [built]
visualization.py Shape-soup renderer (FK-driven), capture, video (--make-videos). [built]
collision.py     Shape-soup collision checker (FK-driven) + allowed pairs.         [built]
ik/              InverseKinematics protocol; NumericalIK (DLS) + IKFastSolver.    [built]
planning/        ConfigurationSpace + MotionPlanner; BiRRTPlanner, OMPLPlanner.  [built]
robots/          Robot composition + Panda, Kinova, TidyBot, Vega (bimanual).    [built]
manipulation/    Primitive ABC; Pick, Place with injected grasp generators.       [planned]
```

The renderer is shape-soup: it creates one PyBullet visual body per node shape
and positions every body from the tree's own forward kinematics, so a grasped
object (re-parented in the tree) renders with no special handling and nothing
relies on PyBullet's articulation. Meshes go through PyBullet's file importer;
the visual loader reads `.obj`/`.stl`/`.dae` directly while `createCollisionShape`
reads only `.obj`/`.stl`, so anything outside the relevant set (e.g. `.glb`, or
`.dae` for collision) is converted to `.obj` once via trimesh and cached on disk.
The collision checker
uses the same per-shape, FK-positioned approach: one collision body per shape,
positioned by FK, with non-ignored body pairs tested via `getClosestPoints`. A
grasped object (re-parented in the tree) collides with the environment for free.
Same-node and tree-adjacent pairs are ignored; rest-overlapping pairs (an
allowed-collision matrix) are discovered with `pairs_in_collision` and supplied
to `ignore`. There is no `CollisionBackend` ABC yet — a planner takes a plain
`config -> bool` callable; the ABC arrives with a second backend (e.g. FCL).

A `JointSpace` is a group of actuated joints in a fixed order, supplying the
sample/distance/interpolate operations a sampling planner needs plus conversion
between a flat coordinate vector and a `Configuration`. Continuous joints
(revolute with infinite limits, e.g. Kinova's joints 1/3/5/7) are handled
throughout: they sample over `[-pi, pi]`, and distance and interpolation take
the shorter way around the 2*pi seam. A `JointSpace` is one implementation of
the `ConfigurationSpace` protocol (sample/distance/interpolate plus vector<->
config conversion); `SE2Space` is another, for a planar mobile base whose `yaw`
wraps like a continuous joint. `BiRRTPlanner` adapts the generic `BiRRT` from
`prpl_utils` to any `ConfigurationSpace`, taking a plain `config -> bool`
collision callable (the checker satisfies this) and holding non-planned joints
fixed at the start configuration -- so the same planner does joint-space arm
planning and SE(2) base planning. Both planners satisfy the `MotionPlanner`
protocol (`plan(start, goal) -> path | None`): `BiRRTPlanner` and `OMPLPlanner`,
which wraps OMPL's `RRTConnect` (its low-level state handling stays inside the
class, and a `ConfigurationSpace.bounds()` gives OMPL a finite sampling range).
BiRRT has lower overhead on easy problems; OMPL scales far better through narrow
passages (see `notebooks/motion_planner_comparison.ipynb`). Robots are
composition over a tree:
named groups (a `JointSpace` arm, an `SE2Space` base) plus `Manipulator`s, each
pairing an EE frame and an IK solver with a group. A single-arm robot has one
manipulator; a bimanual robot (Vega) has two, each with its own IK; a mobile
base is just an extra group. No inheritance tower -- a mobile base is a
`PlanarJoint` edge from the root, so the whole mobile manipulator is one tree,
and robot-specific IK (Vega's EAIK joint-locking solver) plugs in via the
`InverseKinematics` protocol.

IK is structured so that robot-specific solvers are first-class: the
`InverseKinematics` protocol is just `solve(target_pose, seed) -> Configuration
| None`, and any conforming object (generic or bespoke) is usable
interchangeably. Two implementations ship. `NumericalIK` reuses the same
`JointSpace` and solves via damped-least-squares differential steps
(`dq = J^T (J J^T + lambda^2 I)^-1 e`) with a finite-difference Jacobian over
the tree's FK; it is a *local* solver (converges from a seed in the basin).
`IKFastSolver` wraps a per-robot IKFast module (compiled on demand from
committed C++), using the tree's FK to put the target in the solver's base
frame and returning the limit-respecting candidate closest to the seed; it is
*global* and seedless. `follow_end_effector_path` deliberately requires
`NumericalIK`, not the protocol: smooth Cartesian tracking needs continuous
differential stepping, whereas a global solver may jump between IK branches.
Robot-specific analytic solvers that no general method handles (e.g. the
Dexmate Vega's non-spherical wrist) plug in as their own `InverseKinematics`.

A `Robot` is composition over a tree, not an inheritance tower: it carries named
joint groups (`{"arm": JointSpace, "gripper": JointSpace}`), an EE frame, an
injected `InverseKinematics`, a home configuration, and the robot's intrinsic
allowed-collision pairs (discovered from the robot alone, so scene collisions
are never masked). A specific robot is a configured instance from a factory
(`make_panda`); algorithms consume the capabilities they need (a joint group,
the IK solver), so robots with extra capabilities just expose more -- a mobile
base adds an SE(2) space, a bimanual robot a second arm group -- and stay
swappable without a class hierarchy. The uniform planning substrate makes this
work: `BiRRTPlanner` already consumes anything offering sample/distance/
interpolate, so base SE(2) planning and arm joint planning are the same planner
over different spaces. The `ConfigurationSpace` protocol and an `SE2Space` are
extracted when the first mobile robot (TidyBot) needs them.

## What is built today

The minimal, fully-tested core:

- `geometry.transforms` — `pose_from_pybullet`, `pose_to_pybullet`.
- `tree.joints` — `Joint` ABC + `FixedJoint` (0 DOF), `RevoluteJoint` (1),
  `PrismaticJoint` (1), `PlanarJoint` (3). Each maps joint values to an `SE3`.
- `tree.kinematic_tree` — `Node`, `Edge`, `KinematicTree` (`add_node`,
  `add_edge`, `forward_kinematics`, `relative_pose`, `attach`,
  `actuated_joint_names`, `joint`, `path_from_root`).
- `tree.state` — `KinematicState` snapshot of actuated joint values.
- `geometry.shapes` — `MeshShape`/`BoxShape`/`CylinderShape`/`SphereShape`.
- `loading.urdf` — `load_urdf` (yourdfpy → tree, with per-node geometry).
- `meshes` — `to_pybullet_mesh` (native passthrough + cached `.glb`→`.obj`).
- `visualization` — `PyBulletRenderer`, `render_configurations`, `capture_image`,
  `save_video`, plus the `--make-videos` test fixture.
- `collision` — `PyBulletCollisionChecker` (`in_collision`, `pairs_in_collision`,
  `ignore`).
- `planning` — `ConfigurationSpace` protocol (`JointSpace`, `SE2Space`) and
  `MotionPlanner` protocol with `BiRRTPlanner` (`prpl_utils.BiRRT`) and
  `OMPLPlanner` (OMPL `RRTConnect`).
- `ik` — `InverseKinematics` protocol, `NumericalIK` (Jacobian-DLS differential
  solve), `IKFastSolver` (per-robot analytic solve), and
  `follow_end_effector_path` (warm-started Cartesian-path tracking).
- `robots` — `Robot` (composition: named groups, EE, injected IK, home,
  intrinsic ACM) with `make_panda`, `make_kinova` (Gen3, continuous joints),
  `make_tidybot` (Gen3 on an SE(2) mobile base), and `make_vega` (bimanual, two
  manipulators with a bespoke EAIK `VegaArmIK`).

Tests cover conversions, every joint type, FK propagation, grasp re-parenting,
snapshotting, URDF loading, FK equivalence with PyBullet on Panda, `.glb` mesh
conversion, shape-soup rendering, collision checking (primitives, adjacency,
attached-object-vs-environment, and Panda rest pose), motion planning (BiRRT and
OMPL both conforming to `MotionPlanner`, joint-space geometry, continuous-joint
wrap-around, steering around an obstacle, and a Panda plan around a block), IK
(numerical reaching from a nearby seed, smooth
warm-started EE-path following, IKFast global solves on Panda, branch selection,
unreachable targets, and both solvers conforming to the `InverseKinematics`
protocol), SE(2) spaces (workspace-bounded sampling, yaw wrap-around,
interpolation), and robot assembly (Panda/Kinova/TidyBot groups/EE/IK/home/ACM,
IK through the robot, a self-collision-free home, the Gen3's continuous joints,
planning the arm around an obstacle, driving TidyBot's SE(2) base around a
floor pillar with the same `BiRRTPlanner`, and Vega's two arms each solving IK
through their own manipulator).

## Milestones (each adds code + tests)

1. **Geometry + tree core** — done.
2. **Loading + visualization** — done. URDF → tree with geometry (FK validated
   against PyBullet); shape-soup FK-driven renderer + `--make-videos` pipeline.
3. **Collision** — done. Shape-soup `PyBulletCollisionChecker` (FK-positioned
   bodies, allowed-pair discovery); a grasped object collides for free.
4. **IK** — done. `InverseKinematics` protocol with two implementations:
   `NumericalIK` (Jacobian-DLS) + `follow_end_effector_path` (the warm-started
   ee-follow jitter fix), and `IKFastSolver` (per-robot analytic, compiled on
   demand). Robot-specific solvers (e.g. Vega EAIK) plug in via the protocol.
5. **Planning** — done. `ConfigurationSpace` protocol with `JointSpace` and
   `SE2Space`, and `BiRRTPlanner` (wrapping `prpl_utils`) over either via a
   `config -> bool` callable. `OMPLPlanner` next, extracting a `MotionPlanner`
   ABC once a second planner exists.
6. **Robots** — done. `make_panda`, `make_kinova` (Gen3), `make_tidybot` (Gen3
   on an SE(2) mobile base), and `make_vega` (bimanual). Each robot exposes
   *explicitly named* groups (e.g. `"arm"`, `"base"`, `"left_arm"`) plus
   `Manipulator`s (EE frame + IK), home, and intrinsic ACM — never a single
   ambiguous "the robot's joints" vector, and the gripper is a group, not a
   `finger_state` scalar (principle 7). TidyBot added `SE2Space` and the
   `ConfigurationSpace` protocol; Vega added the `Manipulator` mapping (two
   arms) and a bespoke EAIK solver via the `InverseKinematics` protocol.
7. **OMPL** — done. `OMPLPlanner` wraps OMPL's `RRTConnect` over the existing
   `ConfigurationSpace`, behind the extracted `MotionPlanner` protocol.
   `notebooks/motion_planner_comparison.ipynb` compares it with BiRRT (timing,
   rendered plans, narrow-passage scaling).
8. **Manipulation** — `Pick`/`Place` on the new stack.
