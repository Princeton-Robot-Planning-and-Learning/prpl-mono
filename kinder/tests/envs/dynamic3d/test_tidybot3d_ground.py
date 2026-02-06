"""Tests for the ground regions: site creation, placement, and sizing."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kinder.envs.dynamic3d.objects.base import MujocoGround


def test_ground_region_site_creation_and_placement():
    """Test ground construction with regions.

    Verify site creation, placement, and sizing.

    This test verifies that:
    1. All regions are created successfully
    2. Sites are created for each region
    3. Sites are positioned above ground (z > 0)
    4. Sites do NOT extend below ground (z_min >= 0)
    5. Sites span from ground surface to ground_placement_threshold * 2 above
    6. Sites have correct dimensions matching region bounds with placement threshold
    7. All sites are added to the worldbody XML element
    """
    # Create ground fixture config with multiple regions
    regions_config = {
        # Small region in the center
        "center_region": {
            "ranges": [[-0.1, -0.1, 0.1, 0.1]],
            "rgba": [1.0, 0.0, 0.0, 0.3],
        },
        # Larger region to the right
        "right_region": {
            "ranges": [[0.15, -0.2, 0.5, 0.2]],
            "rgba": [0.0, 1.0, 0.0, 0.3],
        },
        # Region with multiple sub-ranges
        "multi_region": {
            "ranges": [[-0.3, -0.4, -0.1, -0.2], [0.1, 0.2, 0.3, 0.4]],
            "rgba": [0.0, 0.0, 1.0, 0.3],
        },
        # Tall region (should extend above ground)
        "tall_region": {
            "ranges": [[-0.5, 0.3, -0.2, 0.6]],
            "rgba": [1.0, 1.0, 0.0, 0.3],
        },
    }

    # Create a mock worldbody element to verify sites are added to it
    worldbody = ET.Element("worldbody")

    # Create the ground fixture with worldbody
    ground = MujocoGround(regions=regions_config, worldbody=worldbody)

    # Verify that all regions were created
    assert (
        len(ground.region_objects) == 4
    ), f"Expected 4 region groups, got {len(ground.region_objects)}"
    for region_name in regions_config:
        assert (
            region_name in ground.region_objects
        ), f"Region {region_name} not found in ground regions"

    # Test each region
    test_cases = [
        {
            "region_name": "center_region",
            "expected_num_regions": 1,
            "ranges": [[-0.1, -0.1, 0.1, 0.1]],
        },
        {
            "region_name": "right_region",
            "expected_num_regions": 1,
            "ranges": [[0.15, -0.2, 0.5, 0.2]],
        },
        {
            "region_name": "multi_region",
            "expected_num_regions": 2,
            "ranges": [[-0.3, -0.4, -0.1, -0.2], [0.1, 0.2, 0.3, 0.4]],
        },
        {
            "region_name": "tall_region",
            "expected_num_regions": 1,
            "ranges": [[-0.5, 0.3, -0.2, 0.6]],
        },
    ]

    for test_case in test_cases:
        region_name = test_case["region_name"]
        expected_num_regions = test_case["expected_num_regions"]
        ranges = test_case["ranges"]

        # Get the region objects
        regions = ground.region_objects[region_name]
        assert len(regions) == expected_num_regions, (
            f"Expected {expected_num_regions} regions for {region_name}, "
            f"got {len(regions)}"
        )

        # Check each region/sub-range
        for region_idx, region in enumerate(regions):
            # Verify site element exists
            assert (
                region.site_element is not None
            ), f"Site element should exist for {region_name} region {region_idx}"

            site_name = region.site_element.get("name", "")
            assert (
                site_name != ""
            ), f"Site name should not be empty for {region_name} region {region_idx}"

            # Extract site position and size from XML
            site_pos_str = region.site_element.get("pos", "")
            site_size_str = region.site_element.get("size", "")

            assert site_pos_str, f"Site {site_name} has no position"
            assert site_size_str, f"Site {site_name} has no size"

            site_pos = [float(x) for x in site_pos_str.split()]
            site_size = [float(x) for x in site_size_str.split()]

            assert (
                len(site_pos) == 3
            ), f"Site position should have 3 components, got {len(site_pos)}"
            assert (
                len(site_size) == 3
            ), f"Site size should have 3 components, got {len(site_size)}"

            site_x, site_y, site_z = site_pos
            size_x, size_y, size_z = site_size

            # Get the corresponding range for this sub-region
            x_start, y_start, x_end, y_end = ranges[region_idx]

            # Verify site X-Y position is at the center of the range
            expected_center_x = (x_start + x_end) / 2
            expected_center_y = (y_start + y_end) / 2

            assert np.isclose(site_x, expected_center_x, atol=1e-6), (
                f"Site X position mismatch for {region_name} region {region_idx}: "
                f"expected {expected_center_x}, got {site_x}"
            )
            assert np.isclose(site_y, expected_center_y, atol=1e-6), (
                f"Site Y position mismatch for {region_name} region {region_idx}: "
                f"expected {expected_center_y}, got {site_y}"
            )

            # Verify site Z position is above ground (centered between 0 and 2*threshold)
            # For 4-value ranges, z_start=0 and z_end=ground_placement_threshold
            # bbox z range is [max(0, z_start-threshold), z_end+threshold]
            # = [0, 2*threshold]
            # The site is centered between 0 and 2*ground_placement_threshold
            ground_placement_threshold = 0.05  # Actual threshold value
            expected_z = ground_placement_threshold  # Center of [0, 0.1]
            assert np.isclose(site_z, expected_z, atol=1e-6), (
                f"Site Z position should be at {expected_z} "
                f"for {region_name} region {region_idx}, got {site_z}"
            )

            # Verify site region spans correct Z range (entirely above ground)
            # The site extends from z - size_z to z + size_z
            z_min = site_z - size_z
            z_max = site_z + size_z

            # Site should span from z=0 (surface) to z=0.1 (2*threshold above)
            # bbox z range is [0, 0.1]
            # z_center = (0 + 0.1) / 2 = 0.05
            # z_size = (0.1 - 0) / 2 = 0.05
            # So z_min = 0.05 - 0.05 = 0, z_max = 0.05 + 0.05 = 0.1
            expected_z_min = 0.0  # At ground surface
            expected_z_max = 2 * ground_placement_threshold  # Above ground

            assert np.isclose(z_min, expected_z_min, atol=1e-6), (
                f"Site {site_name} Z minimum mismatch for {region_name} "
                f"region {region_idx}: "
                f"expected {expected_z_min}, got {z_min}. Site must not go below ground."
            )
            assert np.isclose(z_max, expected_z_max, atol=1e-6), (
                f"Site {site_name} Z maximum mismatch for {region_name} "
                f"region {region_idx}: "
                f"expected {expected_z_max}, got {z_max}"
            )

            # Verify site doesn't go below ground
            assert z_min >= 0.0, (
                f"Site {site_name} extends below ground (z_min={z_min}). "
                f"Sites must not go below ground."
            )

            # Verify site X-Y sizes match range spans (with threshold added)
            expected_size_x = (x_end - x_start + 2 * ground_placement_threshold) / 2
            expected_size_y = (y_end - y_start + 2 * ground_placement_threshold) / 2

            assert np.isclose(size_x, expected_size_x, atol=1e-6), (
                f"Site X size mismatch for {region_name} region {region_idx}: "
                f"expected {expected_size_x}, got {size_x}"
            )
            assert np.isclose(size_y, expected_size_y, atol=1e-6), (
                f"Site Y size mismatch for {region_name} region {region_idx}: "
                f"expected {expected_size_y}, got {size_y}"
            )

            # Verify site Z size is correct (half of 2*ground_placement_threshold)
            expected_size_z = ground_placement_threshold
            assert np.isclose(size_z, expected_size_z, atol=1e-6), (
                f"Site Z size mismatch for {region_name} region {region_idx}: "
                f"expected {expected_size_z}, got {size_z}"
            )

            # Verify RGBA values
            rgba_str = region.site_element.get("rgba", "")
            assert rgba_str, f"Site {site_name} has no RGBA value"
            rgba_values = [float(x) for x in rgba_str.split()]
            expected_rgba = regions_config[region_name]["rgba"]
            assert (
                len(rgba_values) == 4
            ), f"RGBA should have 4 components, got {len(rgba_values)}"
            for i, (actual, expected) in enumerate(zip(rgba_values, expected_rgba)):
                assert np.isclose(actual, expected, atol=1e-6), (
                    f"RGBA component {i} mismatch for {region_name} "
                    f"region {region_idx}: "
                    f"expected {expected}, got {actual}"
                )

    # Verify that all sites were added to the worldbody
    worldbody_sites = worldbody.findall("site")
    expected_site_count = sum(len(v["ranges"]) for v in regions_config.values())

    assert (
        len(worldbody_sites) == expected_site_count
    ), f"Expected {expected_site_count} sites in worldbody, found {len(worldbody_sites)}"

    # Verify each site in worldbody has proper attributes
    for site in worldbody_sites:
        site_name = site.get("name", "")
        assert site_name.startswith(
            "ground_"
        ), f"Site name should start with 'ground_', got {site_name}"
        assert site.get("type") == "box", f"Site {site_name} should be of type 'box'"
        assert site.get("pos"), f"Site {site_name} should have position"
        assert site.get("size"), f"Site {site_name} should have size"
        assert site.get("rgba"), f"Site {site_name} should have RGBA values"


def test_ground_sample_pose_in_region():
    """Test that sampling poses from ground regions produces valid positions."""
    regions_config = {
        "region1": {
            "ranges": [[0.0, 0.0, 1.0, 1.0]],
            "rgba": [1.0, 0.0, 0.0, 0.3],
        },
    }

    ground = MujocoGround(regions=regions_config)
    np_random = np.random.default_rng(seed=42)

    # Sample multiple poses
    for _ in range(10):
        x, y, z, yaw = ground.sample_pose_in_region("region1", np_random)

        # Verify position is within the region
        assert 0.0 <= x <= 1.0, f"X position {x} outside region bounds"
        assert 0.0 <= y <= 1.0, f"Y position {y} outside region bounds"

        # Verify Z is within the default ground placement range
        # For 4-value ranges (no explicit z), z is sampled from
        # [0, 2*ground_placement_threshold]
        # The expected z value is at the center of this range
        z_min = 0.0
        z_max = 2 * ground.ground_placement_threshold
        assert z_min <= z <= z_max, (
            f"Z position {z} outside default ground placement range "
            f"[{z_min}, {z_max}]"
        )

        # Verify yaw is in valid range
        assert 0.0 <= yaw <= 2 * np.pi, f"Yaw {yaw} outside valid range [0, 2π]"


def test_ground_check_in_region():
    """Test ground region containment checking."""
    regions_config = {
        "test_region": {
            "ranges": [[0.0, 0.0, 1.0, 1.0]],
            "rgba": [1.0, 0.0, 0.0, 0.3],
        },
    }

    ground = MujocoGround(regions=regions_config)

    # Position at ground level within region bounds
    # Note: region extends by ground_placement_threshold, so the actual region bounds are
    # [-0.1, -0.1] to [1.1, 1.1] in XY
    pos_inside = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    assert ground.check_in_region(
        pos_inside, "test_region"
    ), "Position inside region should be detected"

    # Position outside region bounds
    pos_outside = np.array([2.0, 2.0, 0.0], dtype=np.float32)
    assert not ground.check_in_region(
        pos_outside, "test_region"
    ), "Position outside region should not be detected"


def test_ground_region_invalid_region_name():
    """Test that requesting invalid region names raises errors."""
    regions_config = {
        "valid_region": {
            "ranges": [[0.0, 0.0, 1.0, 1.0]],
            "rgba": [1.0, 0.0, 0.0, 0.3],
        },
    }

    ground = MujocoGround(regions=regions_config)
    np_random = np.random.default_rng()

    with pytest.raises(KeyError):
        ground.sample_pose_in_region("nonexistent_region", np_random)

    with pytest.raises(ValueError):
        pos = np.array([0.5, 0.5, 0.005], dtype=np.float32)
        ground.check_in_region(pos, "nonexistent_region")
