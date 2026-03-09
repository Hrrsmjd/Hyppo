## Reading Training Metrics

Each heartbeat includes run summaries plus metric history for active and
recent completed runs. The most important fields are:

- `best_<metric>`
- `latest_<metric>`
- `latest_train_loss`
- `elapsed_time_seconds`
- `progress_percent`
- `trend`

The primary metric and direction come from the config:
- if the objective is `minimize`, lower primary-metric values are better
- if the objective is `maximize`, higher primary-metric values are better

Use the primary metric comparatively, not in isolation. Prefer
comparing runs at similar progress percentages rather than raw
wall-clock time.

- **Primary metric improving steadily**: healthy training; usually keep
  running
- **Primary metric improving but slowly**: often still worth keeping if
  the run is competitive at similar progress
- **Train loss down, validation loss flat or rising**: likely
  overfitting; regularization or augmentation may need to change
- **Oscillation with weak net improvement in the primary metric**:
  learning rate may be too high or batch size too small
- **Very poor best primary metric after substantial progress**: likely
  not worth more budget
- **Best run sits near the edge of a range**: consider expanding that
  dimension in the search space

## Using Metrics for Search Space Decisions

The metrics tables should drive explicit decisions:

- If many runs show overfitting, add or emphasize regularization
  parameters if they are allowed.
- If high-learning-rate runs repeatedly oscillate or diverge, narrow the
  upper end of the learning-rate range.
- If low-learning-rate runs improve too slowly while staying stable,
  narrow the lower end upward.
- If a parameter changes widely across runs with little effect on
  outcomes, remove it or narrow it sharply.
- If one region dominates, focus future runs there but keep enough
  diversity to test whether the pattern is real.
