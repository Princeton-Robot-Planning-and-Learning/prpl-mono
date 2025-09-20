# pybullet-helpers

Some utility functions for PyBullet. Copied and modified from [predicators](https://github.com/Learning-and-Intelligent-Systems/predicators), which in turn was heavily based on the pybullet-planning repository by Caelan Garrett (https://github.com/caelan/pybullet-planning/). In addition, the structure is loosely based off the pb_robot repository by Rachel Holladay (https://github.com/rachelholladay/pb_robot). [Will Shen](https://shen.nz/) made huge contributions.

## Requirements

- Python 3.10+
- Tested on MacOS Catalina

## Installation

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

```
# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.

## Adding New Robots

To add a new robot, build off an existing example.
For inverse kinematics, PyBullet's IK solver will be used by default.
It is not very good.
IKFast is much better, but then you need to compile robot-specific IK models.
This process needs to be automated further, but here is some guidance:
1. Install Docker on an Ubuntu machine. (You will only need Ubuntu to compile once; IKFast should work cross-platform.)
2. Follow the instructions on [pyikfast](https://github.com/cyberbotics/pyikfast).
3. Save the `cpp` file that is generated. You won't need the other files.
4. Make a new directory in this repository inside `third_party/ikfast`. Copy in the `cpp` file and rename it to match the existing examples (e.g., `ikfast_panda_arm.cpp`.)
5. Modify the `cpp` file in two ways: (1) Add `#include "Python.h"` at the top; (2) add python bindings at the bottom (copy and change the robot name from an existing example like `ikfast_panda_arm.cpp`).
6. Copy in the other files from an example directory like `third_party/ikfast/panda_arm`. Modify `robot_name` in `setup.py`.
7. Add `IKInfo` inside your new robot class in `robots/`.

Contributions are welcome to improve this process, especially steps 3 onward.

*Note for Robot URDFs:
For consistency with Bullet IK, ensure that the inertial frame of the robot URDF's base link is not offset from the its link frame. If that's necessary, a possible workaround is to add a dummy base link to the URDF and connecting this to the real base link via a fixed joint.

```
<link name="dummy_base" />

<joint name="dummy_joint" type="fixed">
    <parent link="dummy_base"/>
    <child link="panda_link0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```
