#!/usr/bin/env python
"""Quick script to inspect HDF5 file structure and shapes."""

import argparse

import h5py  # type: ignore


def print_hdf5_structure(name: str, obj) -> None:
    """Print HDF5 object name and shape if it's a dataset."""
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        # Print attributes if any
        if obj.attrs:
            attrs = dict(obj.attrs)
            print(f"  {name}/ (attrs: {attrs})")
        else:
            print(f"  {name}/")


def main() -> None:
    """Main function to inspect HDF5 file structure."""
    parser = argparse.ArgumentParser(description="Inspect HDF5 file structure")
    parser.add_argument(
        "hdf5_path",
        type=str,
        help="Path to HDF5 file",
    )
    parser.add_argument(
        "--max_demos",
        type=int,
        default=3,
        help="Maximum number of demos to show in detail (default: 3)",
    )
    args = parser.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        print(f"\n=== HDF5 File: {args.hdf5_path} ===\n")

        # Print top-level structure
        print("Top-level keys:", list(f.keys()))
        print()

        # Print attributes at root level
        if f.attrs:
            print("Root attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print()

        # If there's a 'data' group, explore it
        if "data" in f:
            data_group = f["data"]
            print("data/ group:")

            # Print data group attributes
            if data_group.attrs:
                print("  Attributes:")
                for key, value in data_group.attrs.items():
                    print(f"    {key}: {value}")
                print()

            # List demos
            demo_keys = sorted(
                [k for k in data_group.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
            print(f"  Number of demos: {len(demo_keys)}")
            print()

            # Show structure of first few demos
            for _, demo_key in enumerate(demo_keys[: args.max_demos]):
                demo = data_group[demo_key]
                print(f"  {demo_key}/")

                # Print demo attributes
                if demo.attrs:
                    for key, value in demo.attrs.items():
                        print(f"    (attr) {key}: {value}")

                # Print datasets in this demo
                for item_name in demo.keys():
                    item = demo[item_name]
                    if isinstance(item, h5py.Dataset):
                        print(
                            f"    {item_name}: shape={item.shape}, dtype={item.dtype}"
                        )
                    elif isinstance(item, h5py.Group):
                        print(f"    {item_name}/ (group with {len(item.keys())} items)")
                        # Optionally show contents of nested group
                        for nested_name in item.keys():
                            nested = item[nested_name]
                            if isinstance(nested, h5py.Dataset):
                                print(
                                    f"      {nested_name}: shape={nested.shape}, dtype={nested.dtype}"  # pylint: disable=line-too-long
                                )
                            else:
                                print(f"      {nested_name}/ (group)")
                print()

            if len(demo_keys) > args.max_demos:
                print(f"  ... and {len(demo_keys) - args.max_demos} more demos")
                print()

        else:
            # Generic traversal if no 'data' group
            print("Full structure:")
            f.visititems(print_hdf5_structure)


if __name__ == "__main__":
    main()
