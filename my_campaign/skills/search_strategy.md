## Hyperparameter Search Strategy

You are optimizing hyperparameters for a transformer model. Your budget
is limited, so be strategic.

### Initialization (First Heartbeat)
Read the model description and available hyperparameters in config.json.
Based on your knowledge of the architecture and task, define an initial
search space using the `initialize_search_space` tool. Consider:
- What are typical good ranges for this architecture?
- Which hyperparameters are most likely to impact performance?
- Start broader than you think necessary — you can narrow later.
Don't include every available parameter. Start with the 4-6 most
impactful ones and add more if needed.

### Phase 1 — Coarse Sweep (first 20% of budget)
Launch diverse configurations spanning the search space. Vary multiple
parameters at once. Goal: map the landscape, find promising regions,
and identify which parameters matter.

After the coarse sweep, update the search space:
- Remove parameters that don't seem to affect performance
- Narrow ranges around promising regions
- Add new parameters if you suspect they'd help (e.g., add weight_decay
  if you're seeing overfitting)

### Phase 2 — Narrowing (middle 60% of budget)
Focus on regions around the best runs. Vary 1-2 parameters at a time.
Continue refining the search space as you learn more. This is where
most of the gains happen.

### Phase 3 — Fine-tuning (final 20% of budget)
Small perturbations of the top 2-3 configurations. Confirm results
are stable. Search space should be tight around the best-known values.

### Search Space Management
- Always document why you're changing the search space (use the notes
  and changelog fields)
- If the best result is at the boundary of a range, expand that range
- If a large region of the space consistently performs poorly, narrow it
- Be willing to add parameters mid-campaign if evidence suggests they
  matter (e.g., add label_smoothing if you're overfitting)
- Be willing to remove parameters that don't affect results — this
  focuses the remaining budget on what matters
