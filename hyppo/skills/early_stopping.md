## When to Kill a Run

Kill a run early only when that clearly improves the remaining budget.

### Kill immediately if:

- The primary metric has gone invalid (NaN) or clearly catastrophic
  relative to the current best after enough progress to trust the signal
- The run is failing in a way that is unlikely to recover

### Consider killing if:

- At similar progress, the run is materially worse than the current best
  and its trend is flat or deteriorating
- Training loss is dropping but validation loss is rising, suggesting
  strong overfitting
- Recent primary-metric history shows repeated large oscillations with
  no net improvement
- The run is in a crowded region of the space and provides little new
  information

### Keep running if:

- The primary metric is still moving in the favorable direction and the
  run remains plausible at its current progress
- You're in the fine-tuning phase and need to confirm a result
- The run is testing a genuinely novel region of the search space
- The run is behind on wall-clock time but still competitive by
  `progress_percent`

### Early-Stopping Discipline

- Compare runs at similar `progress_percent` whenever possible.
- Do not kill solely because a run started worse than the current best.
- Be more conservative early in a campaign when you still need
  information about the landscape.
- Be more aggressive late in a campaign when the budget should focus on
  the best regions.
