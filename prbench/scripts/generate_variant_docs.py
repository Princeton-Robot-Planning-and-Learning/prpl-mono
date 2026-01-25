"""Generate observation and action space tables for all environment variants.

Usage:
  python generate_variant_docs.py                    # Generate docs for all variants
  python generate_variant_docs.py --force            # Force regenerate all variants
  python generate_variant_docs.py --variant Motion2D-p5  # Generate docs for specific variant
"""

import argparse
from pathlib import Path

import prbench

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "envs" / "variants"


def sanitize_env_id(env_id: str) -> str:
    """Remove prbench/ prefix and version suffix from env ID."""
    assert env_id.startswith("prbench/")
    env_id = env_id[len("prbench/") :]
    env_id = env_id.replace("/", "_")
    assert env_id[-3:-1] == "-v"
    return env_id[:-3]


def generate_variant_markdown(variant_id: str) -> str:
    """Generate markdown with observation and action tables for a variant."""
    env = prbench.make(variant_id, render_mode="rgb_array")
    sanitized_id = sanitize_env_id(variant_id)

    md = f"# {sanitized_id}\n\n"
    md += f"Variant ID: `{variant_id}`\n\n"
    md += "## Observation Space\n"
    md += env.metadata["observation_space_description"] + "\n\n"
    md += "## Action Space\n"
    md += env.metadata["action_space_description"] + "\n"

    env.close()
    return md


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate observation and action tables for environment variants"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all variants"
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Generate docs for a specific variant (e.g., Motion2D-p5)",
    )
    args = parser.parse_args()

    print("Generating variant documentation...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prbench.register_all_environments()

    env_classes = prbench.get_env_classes()

    all_variants = []
    for class_info in env_classes.values():
        all_variants.extend(class_info["variants"])

    if args.variant:
        matching = [v for v in all_variants if args.variant in v]
        if not matching:
            print(f"Error: No variant matching '{args.variant}' found")
            print("Available variants:")
            for v in sorted(all_variants):
                print(f"  {v}")
            return
        all_variants = matching
        print(f"Generating docs for {len(matching)} matching variant(s)")

    total_variants = len(all_variants)
    generated_count = 0

    for variant_id in sorted(all_variants):
        sanitized_id = sanitize_env_id(variant_id)
        output_file = OUTPUT_DIR / f"{sanitized_id}.md"

        if not args.force and output_file.exists():
            print(f"  Skipping {sanitized_id} (already exists)")
            continue

        print(f"  Generating {sanitized_id}...")
        try:
            md = generate_variant_markdown(variant_id)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(md)
            generated_count += 1
        except Exception as e:
            print(f"    Error generating {sanitized_id}: {e}")

    print(f"Finished. Generated {generated_count}/{total_variants} variant docs.")


if __name__ == "__main__":
    _main()
