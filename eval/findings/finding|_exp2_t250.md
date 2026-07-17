# Experiment 2- Candidate A: T=250

## Settings
Same as Experiment 1, with one change: T=250 instead of T=1000 (fewer
diffusion steps). Schedule stays linear, batch_size=32, everything else
unchanged.

## Why this deviation
The paper uses T=1000 diffusion steps (Section 4). Given our compute
budget, we wanted to test whether fewer denoising steps still lets the
model learn something useful in the same wall-clock time - and whether
it changes how fast visible structure shows up during training.

## Result
Ran for 2 hours, reached ~13,000+ steps. At step 1750, samples still
looked like pure random noise. By step 13250, samples showed clearly
larger color regions and a rougher, "brushstroke" texture not a
recognizable painting yet, but visibly more structure than pure noise.

## Analysis
With fewer noise levels to distinguish between (250 instead of 1000),
the model seems to pick up spatial structure faster relative to the
number of training steps it's seen. This is the same batch_size=32 fix
from Experiment 1 (128 still causes OOM on this GPU/architecture).
