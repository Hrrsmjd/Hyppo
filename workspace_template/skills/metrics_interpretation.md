## Reading Loss Curve Plots

Each heartbeat includes loss curve plots for active runs. Use these
to assess training dynamics:

- **Smooth downward curve**: Healthy training, let it continue
- **Oscillating but trending down**: LR may be slightly high, but
  could still converge — monitor for 2-3 more heartbeats
- **Sharp spike then recovery**: Possible data issue or LR too high;
  if it recovers, fine; if spikes repeat, consider killing
- **Flat plateau for many epochs**: Model has converged or is stuck;
  check if val_loss is competitive with best runs
- **Train loss dropping but val_loss rising**: Overfitting — consider
  killing, or note that regularization parameters (dropout, weight_decay,
  label_smoothing) might need to be added to the search space
- **Loss exploded (NaN or very large)**: Kill immediately

A green dashed line shows the best val_loss across all completed runs.
Runs significantly above this line after many epochs should be killed.

## Using Plots for Search Space Decisions

The plots can inform search space updates:
- If many runs show overfitting (train-val gap), consider adding
  regularization parameters
- If runs with high LR all show oscillation, narrow the LR upper bound
- If loss curves look identical across different values of a parameter,
  that parameter probably doesn't matter — remove it
