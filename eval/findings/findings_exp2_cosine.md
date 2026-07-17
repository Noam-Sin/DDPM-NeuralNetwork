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
Ran for the full 2-hour budget but only reached ~9750 steps - noticeably
slower per step than the T=250 and T=1000/linear runs in the same time
window. At step 9750, samples still look like dense, fine-grained
colorful noise, without the larger structure that showed up in the
T=250 candidate at a similar step count.

## Analysis
The slower step rate meant fewer total training steps fit in the same
time budget, which likely explains why less structure had formed by the
time the job was killed. It's not clear from this run alone whether
cosine would eventually produce better results than linear at matched
step counts - only that under our specific time constraint, it made
less visible progress than T=250 did.
