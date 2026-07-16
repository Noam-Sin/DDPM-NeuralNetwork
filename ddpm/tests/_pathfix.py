"""Adds the ddpm/ package directory to sys.path so tests can `import unet`,
`import diffusion`, etc. exactly like train.py / quick_validation.py do,
regardless of the working directory tests are invoked from."""
import os
import sys

_DDPM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DDPM_DIR not in sys.path:
    sys.path.insert(0, _DDPM_DIR)
