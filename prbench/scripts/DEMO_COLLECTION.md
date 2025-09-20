# PRBench Scripts

This directory contains utility scripts for PRBench demonstration collection and video generation.

## Demo Collection with Controller Support

### collect_demos.py

Collect human demonstrations using keyboard/mouse or PS5 DualSense controller.

#### Usage

```bash
python scripts/collect_demos.py <environment_id> [--demo-dir DEMO_DIR]
```

**Arguments:**
- `environment_id`: Environment ID (must start with 'prbench/')
- `--demo-dir`: Directory to save demonstrations (default: demos/)

**Display Settings:**
- Screen size: 1000x600 pixels
- Render FPS: 20 frames per second
- Font size: 24px
- Side panels: 150px each for analog stick controls

Examples:
```bash
python scripts/collect_demos.py prbench/Motion2D-p1-v0
python scripts/collect_demos.py prbench/Obstruction2D-o2-v0 --demo-dir my_demos
python scripts/collect_demos.py prbench/ClutteredRetrieval2D-o10-v0
```

#### PS5 DualSense Controller Support

The script automatically detects connected PS5 controllers and provides intuitive control mapping:

**Analog Sticks:**
- **Left Stick** (Axis 0,1): Rotate robot (yaw control)
- **Right Stick** (Axis 2,3): Move robot base (x, y movement)
- **Deadzone**: 0.1 (small movements are ignored for precision)

**Face Buttons:**
- **X (Cross)** (Button 0): Toggle vacuum gripper on/off
- **Circle** (Button 1): Reset environment (start new demo)
- **Square** (Button 3): Save current demo

**D-Pad:**
- **D-pad Up** (Button 11): Extend robot arm outward
- **D-pad Down** (Button 12): Retract robot arm inward

**Technical Details:**
- Y-axis inversion applied to both sticks for intuitive control
- Deadzone filtering prevents unwanted small movements
- Side panels (150px each) provide space for virtual analog stick displays

#### Auto-Save Behavior

- ✅ **Goal Reached**: Automatically saves demo and resets for next attempt
- ❌ **Episode Failed/Timeout**: Automatically resets without saving

#### Keyboard/Mouse Fallback

If no controller is detected, the system falls back to keyboard and mouse controls:

- **Mouse**: Click and drag virtual analog sticks on screen
- **W/S**: Extend/retract robot arm
- **Space**: Toggle vacuum gripper
- **R**: Reset environment
- **G**: Save demo
- **Q**: Quit

#### Demo Output

Demos are saved as pickle files in the following structure:
```
demos/
├── EnvironmentName/
│   ├── 0/
│   │   └── timestamp.p
│   ├── 1/
│   │   └── timestamp.p
│   └── ...
```

**File Naming:**
- Files are named with Unix timestamps (e.g., `1752189500.p`)
- Each demo uses a sequential seed starting from 0
- Seed increments automatically after each environment reset

**Demo File Contents:**
- `env_id`: Environment identifier
- `timestamp`: Unix timestamp when demo was saved
- `seed`: Environment seed used for this demo
- `observations`: Sequence of environment observations
- `actions`: Sequence of actions taken
- `rewards`: Float rewards received at each step
- `terminated`: Whether episode ended successfully
- `truncated`: Whether episode was cut short (timeout/failure)

**Image Rendering:**
- Environment renders RGB images (uint8, shape HxWx3 or HxWx4)
- Images are scaled to fit center area while maintaining aspect ratio
- Black side panels (150px each) frame the environment view

## Video Generation

### generate_demo_video.py

Convert saved demonstration files into GIF videos.

#### Usage

```bash
python scripts/generate_demo_video.py <demo_file.p> [options]
```

#### Options

- `--output`, `-o`: Custom output path (default: auto-generated in docs/envs/assets/demo_gifs/)
- `--fps`: Frames per second (default: environment's render_fps)
- `--loop`: Number of loops for GIF (0 = infinite, default: 0)

#### Examples

```bash
# Basic usage - generates GIF in docs/envs/assets/demo_gifs/
python scripts/generate_demo_video.py demos/Motion2D-p1/0/1752189500.p

# Custom output path and settings
python scripts/generate_demo_video.py demos/Motion2D-p1/0/1752189500.p \
  --output my_demo.gif \
  --fps 30 \
  --loop 0
```

## Environment Documentation

### generate_env_docs.py

Automatically generates markdown documentation for all registered environments. This script is typically run as part of the pre-commit process.

#### Usage

```bash
python scripts/generate_env_docs.py
```

This creates documentation files in `docs/envs/` with environment descriptions, observation/action spaces, and embedded demo videos.
