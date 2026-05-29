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
ik/              InverseKinematics ABC; NumericalIK (Jacobian-DLS), AnalyticIK.   [planned]
planning/        ConfigurationSpace ABC + MotionPlanner ABC; BiRRT, OMPL.         [planned]
robots/          Assemblies over a tree: SingleArm, Bimanual, MobileBase, MobileManip. [planned]
manipulation/    Primitive ABC; Pick, Place with injected grasp generators.       [planned]
```

The renderer is shape-soup: it creates one PyBullet visual body per node shape
and positions every body from the tree's own forward kinematics, so a grasped
object (re-parented in the tree) renders with no special handling and nothing
relies on PyBullet's articulation. Meshes go through PyBullet's file importer;
`.obj`/`.stl`/`.dae` pass through directly and other formats (e.g. `.glb`) are
converted to `.obj` once via trimesh and cached on disk. The collision checker
uses the same per-shape, FK-positioned approach: one collision body per shape,
positioned by FK, with non-ignored body pairs tested via `getClosestPoints`. A
grasped object (re-parented in the tree) collides with the environment for free.
Same-node and tree-adjacent pairs are ignored; rest-overlapping pairs (an
allowed-collision matrix) are discovered with `pairs_in_collision` and supplied
to `ignore`. There is no `CollisionBackend` ABC yet — a planner takes a plain
`config -> bool` callable; the ABC arrives with a second backend (e.g. FCL).

The `ConfigurationSpace` abstraction (sample/distance/interpolate/is_valid) is
what lets one planner work over joint space, an SE(2) base, or a Cartesian EE
target uniformly. Robots are composition over a tree (named joint groups, an EE
link, a home config), not an inheritance tower.

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

Tests cover conversions, every joint type, FK propagation, grasp re-parenting,
snapshotting, URDF loading, FK equivalence with PyBullet on Panda, `.glb` mesh
conversion, shape-soup rendering, and collision checking (primitives, adjacency,
attached-object-vs-environment, and Panda rest pose).

## Milestones (each adds code + tests)

1. **Geometry + tree core** — done.
2. **Loading + visualization** — done. URDF → tree with geometry (FK validated
   against PyBullet); shape-soup FK-driven renderer + `--make-videos` pipeline.
3. **Collision** — done. Shape-soup `PyBulletCollisionChecker` (FK-positioned
   bodies, allowed-pair discovery); a grasped object collides for free.
4. **IK** — `InverseKinematics` ABC; `NumericalIK` (Jacobian-DLS, folding in the
   ee-follow jitter fix), then `AnalyticIK` (EAIK/IKFast port).
5. **Planning** — `ConfigurationSpace` + `MotionPlanner` ABC; `BiRRTPlanner`
   (wrapping `prpl_utils`), then `OMPLPlanner`.
6. **Robots** — port Panda, Kinova, Dexmate Vega, TidyBot as assemblies.
7. **Manipulation** — `Pick`/`Place` on the new stack.
