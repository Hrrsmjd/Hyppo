## Reading Training Metrics

Each heartbeat includes a metrics table for active runs showing summary
statistics such as epochs completed, best validation loss, current train
loss, recent validation losses, and a trend label. Use these to assess
training:

- **Steady decrease in both losses**: Healthy training, let it continue
- **Oscillating but trending down**: LR may be slightly high, but
  could still converge — monitor for 2-3 more heartbeats
- **Train loss dropping but val_loss rising**: Overfitting — consider
  killing, or note that regularization parameters (dropout, weight_decay,
  label_smoothing) might need to be added to the search space
- **Both losses flat for many epochs**: Converged or stuck; check if
  val_loss is competitive with best runs
- **Loss exploded (very large values)**: Kill immediately
- **Val loss at search space boundary**: Consider expanding that range

## Using Metrics for Search Space Decisions

The metrics tables can inform search space updates:
- If many runs show overfitting (train-val gap widening), consider adding
  regularization parameters
- If runs with high LR all show oscillation, narrow the LR upper bound
- If loss values look identical across different values of a parameter,
  that parameter probably doesn't matter — remove it to focus the budget
- If the best result is at the boundary of a range, expand that range
