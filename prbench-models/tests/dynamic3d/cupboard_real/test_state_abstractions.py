"""Tests for cupboard real state_abstractions.py."""

from prbench_models.dynamic3d.cupboard_real.state_abstractions import CupboardRealStateAbstractor
from prbench_models.dynamic3d.ground.parameterized_skills import PyBulletSim
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
import prbench

def test_cupboard_real_state_abstraction():
    """Tests for CupboardRealStateAbstractor()."""
    prbench.register_all_environments()
    num_objects = 1
    env = prbench.make(f"prbench/TidyBot3D-cupboard_real-o{num_objects}-v0", render_mode="rgb_array")
    sim = ObjectCentricTidyBot3DEnv(scene_type="cupboard_real",
        num_objects=num_objects,
        render_images=False
    )
    abstractor = CupboardRealStateAbstractor(sim)

    # Check state abstraction in the initial state. The robot's hand should be empty
    # and the object should be on the ground.
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    abstract_state = abstractor.state_abstractor(state)

    import ipdb; ipdb.set_trace()