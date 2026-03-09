## Hyperparameter Search Strategy

You are optimizing under a finite run budget. Use runs to learn quickly,
not just to sample more points.

### Core Priorities

1. Preserve budget for decisions that change the plan.
2. Prefer informative comparisons over random coverage.
3. Avoid duplicate or near-duplicate launches unless you are explicitly
   confirming a promising region.
4. Keep the active set diverse when evidence is weak, and focused when
   evidence is strong.

### First Heartbeat

Before launching anything:

- Read the project description, configuration, and allowed
  hyperparameters carefully.
- Write an initial strategy with your beliefs about the most important
  parameters, likely failure modes, and the first experiments.
- Initialize the search space with the 4-8 parameters most likely to
  matter.

Do not start with every allowed hyperparameter unless the budget is very
large. Too many dimensions early usually wastes budget.

### Early Phase

Goal: identify which parameters and scales matter.

- Launch deliberately diverse runs across the search space.
- Vary multiple parameters at once only while mapping the space.
- Include at least one conservative baseline-like configuration if the
  current search space is aggressive.

### Middle Phase

Goal: narrow around promising regions without collapsing too early.

- Focus on the parameters that actually changed outcomes.
- Narrow ranges where evidence is consistent.
- Keep some diversity among active runs so each heartbeat still teaches
  you something.
- Add new parameters only when the observed metrics suggest a clear
  missing degree of freedom.

### Late Phase

Goal: confirm and refine the best configurations.

- Use small perturbations around the best 2-3 configurations.
- Run confirmation trials only when they settle a real uncertainty.
- Stop spending budget on clearly inferior regions.

### Search Space Management

- Every search-space update must have a concrete reason grounded in the
  observed metrics.
- If the best result is near a range boundary, consider expanding it.
- If a parameter has little visible effect across several runs, narrow
  it aggressively or remove it.
- If one region fails repeatedly, narrow away from it.
- If overfitting appears repeatedly, consider adding regularization
  parameters that are allowed but not yet in the search space.

### Launch Discipline

- Do not launch configurations that are trivially close to active runs
  unless you are intentionally confirming a result.
- Respect concurrency as a learning budget, not just a throughput limit.
- If the current heartbeat produced new information, update the strategy
  before deciding what to launch next.
