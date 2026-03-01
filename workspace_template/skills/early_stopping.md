## When to Kill a Run

Kill a run early to free budget for more promising experiments.

### Kill immediately if:
- Loss has exploded (NaN, very large values, or visible in the plot)
- Val_loss is 2x or more worse than the current best after 30%+ of
  max epochs

### Consider killing if:
- Val_loss has plateaued for 5+ epochs with no improvement, and it's
  significantly worse than the best known run
- Training loss is dropping but val_loss is rising (overfitting)
- The loss curve shows repeated large oscillations

### Keep running if:
- Loss is still actively decreasing, even if not yet competitive
- You're in the fine-tuning phase and need to confirm a result
- The run is testing a genuinely novel region of the search space
