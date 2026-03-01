# Imitation Learning Baselines for KinDER

## Installation

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

```
# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e “.[develop]”
```

## Introduction

We provide three implementations for imitation learning baselines. 

### Diffusion Policy (DP): 
Imitation learning over RGB images, trained with ~100 demos per environment.

### DP + Environment States (DPES): 
Same as DP, but with state vectors also provided as input.

### Finetuned VLA:
A pretrained π0.5 VLA finetuned on the same demos as imitation learning.


## Usage

### Data Collection

1: Using a PS5 controller to collect demonstrations for 2D environments. 

```bash
cd prpl-mono/kinder/scripts
python collect_demos.py $ENV_ID
```

2: Using an iPhone web app to collect demonstrations for 3D environments. 

```bash
cd prpl-mono/kinder-models/scripts
python teleop_dynamics3d_prbench.py --env-name $ENV_ID --show-images
```

Specific Steps:

2.1: Download XR Browser (https://apps.apple.com/us/app/xr-browser/id1588029989)
2.2: Run the teleoperation script.
2.3: get the <server-hostname> of your mac/ubuntu by ipconfig getifaddr en0.
2.4: Make sure your iPhone and your Mac/Ubuntu are under the same wifi.
2.5: On your iPhone XR Browser, navigate to <server-hostname>:5000 to open the teleoperation web app
2.6: Press “Start episode” to begin teleoperation and then move your iPhone


3: Using a VR headset to collect demonstrations for 3D environments. 

TODO

4: Using bilevel planning and parameterized skills to collect demonstrations. 

```bash
cd prpl-mono/kinder-ds-policies/experiments
python experiments/collect_demos_ds.py env=base_motion3d seed=0
```

### Data Format Transfer to hdf5

```bash
cd prpl-mono/kinder-models/scripts
python demos_to_hdf5.py --teleop_data_dir $YOUR_DATA_DIR --output_path $OUTPUT_HDF5_PATH --render_images
```

### Training


1： Clone the diffusion policy repo: 

```bash
git clone git@github.com:Princeton-Robot-Planning-and-Learning/kinder-diffusion-policy.git
```

2： Create a conda/mamba environment following the instructions on kinder-diffusion-policy. 

3： Run training

```bash
cd ~/kinder-diffusion-policy
mamba activate robodiff
python train.py --config-name=train_sweep3d_image
```

### Inference/Evaluations

1: Download checkpoint path or use your own checkpoint

2: Run policy server

```bash
cd ~/kinder-diffusion-policy
mamba activate robodiff
python policy_server.py --ckpt-path $Checkpoint_path 
```

3: Start environment

```bash
cd prpl-mono/kinder-models/scripts
python inference_geom2d.py --env-name $ENV_ID --save-videos --num-seeds 1 --num-episodes 5 --max-steps 200
```

### Troubleshooting

**If `av` installation fails on macOS**: The `av` package requires FFmpeg libraries. If you see pkg-config errors, install FFmpeg via Homebrew and set the PKG_CONFIG_PATH:

```bash
brew install ffmpeg
PKG_CONFIG_PATH=“/opt/homebrew/opt/ffmpeg/lib/pkgconfig” uv pip install av==15.1.0 --no-binary av
```