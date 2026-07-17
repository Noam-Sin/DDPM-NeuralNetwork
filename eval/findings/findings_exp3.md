# Experiment 2- Candidate B: Cosine schedule

## Settings
Same as Experiment 1, with one change: cosine noise schedule instead of
linear. T stays at 1000, batch_size=32, everything else unchanged.

## Why this deviation
Cosine schedules were introduced in later work (Nichol & Dhariwal,
"Improved DDPM", 2021) as a way to improve on the paper's original
linear schedule, with reported benefits especially at low resolutions -
relevant here since we're working at 64x64.

## Result
Ran for the 2-hour budget, reached ~14,250 steps. At step 250, pure
noise as expected. By step 14250, samples still look like dense,
chaotic, fine-grained color mixing - **less** defined than what
Experiment 2's T=250 run showed at its highest step (~13250), which had
clearer, larger color regions.

## Analysis
The trend from Experiment 2 does NOT continue monotonically. Going from
T=250 down to T=100 made the samples look less structured, not more -
the opposite of what we expected going in. A likely explanation: with
only 100 denoising steps, each step has to remove a much larger chunk
of noise at once, making the denoising task harder for the model to
learn well in our limited training budget. This suggests there's a
"sweet spot" around T=250 for this setup, rather than "lower T always
helps" - a more interesting and specific finding than a simple trend.