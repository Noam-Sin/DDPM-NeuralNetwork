# Experiment 1- Baseline (T=1000)

## Settings
Same as the paper (Ho, Jain, Abbeel 2020, Section 4): T=1000, linear
schedule, Adam lr=2e-4, EMA=0.9999. Forced changes due to
our time/compute limits: image_size=64 instead of 32/256, batch_size=32
instead of 128 (see below), and training capped at 2 hours instead of
800,000 iterations.

## Hypothesis
Given only a couple of hours instead of days, we expected the loss to
drop and stabilize, but the samples to still look far from an actual
painting.

## Result
Loss dropped from ~1.0 at step 1 to ~0.03-0.1 within a few hundred
steps, then stayed roughly flat for the rest of the run (~12,000-15,000
steps in 2 hours).

Comparing step 500 to step 12000, the samples barely changed. Both are
dense, colorful, pixel-level noise - no real color blobs or structure
forming. Maybe slightly less contrast by the end, but nothing that
looks like it's turning into a painting.

## Analysis
We initially tried batch_size=128 (matching the paper) and hit a CUDA
Out of Memory error on our 24GB GPU - the model architecture at
image_size=64 with base_channels=128 just doesn't fit at that batch
size. Dropped it to 32 and the run went through fine. Not something we
planned for going in, but worth documenting as a real constraint we hit.

The fact that samples barely changed between step 500 and step 12000
makes sense once you compare the numbers: 12,000 steps is under 2% of
the paper's 800,000. At T=1000, the model just doesn't get enough
gradient updates in our time budget to visibly denoise anything.
