# Imitation Learning Baselines for KinDER

## Installation

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

```
# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```

### Troubleshooting

**If `av` installation fails on macOS**: The `av` package requires FFmpeg libraries. If you see pkg-config errors, install FFmpeg via Homebrew and set the PKG_CONFIG_PATH:

```bash
brew install ffmpeg
PKG_CONFIG_PATH="/opt/homebrew/opt/ffmpeg/lib/pkgconfig" uv pip install av==15.1.0 --no-binary av
```
