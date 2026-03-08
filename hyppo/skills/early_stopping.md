## When to Kill a Run

Kill a run early to free budget for more promising experiments.

### Kill immediately if:
- Loss has exploded (NaN or very large values)
- Val_loss is 2x or more worse than the current best after 30%+ of
  max epochs

### Consider killing if:
- Val_loss has plateaued for 5+ epochs with no improvement, and it's
  significantly worse than the best known run
- Training loss is dropping but val_loss is rising (overfitting)
- Recent val_loss values show repeated large oscillations

### Keep running if:
- Loss is still actively decreasing, even if not yet competitive
- You're in the fine-tuning phase and need to confirm a result
- The run is testing a genuinely novel region of the search space
