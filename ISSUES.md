# Open Issues

## Prioritized Triage (bug-first)

### Fix now (runtime correctness / data integrity)

- **#13** Claude 429 rate-limit failures from oversized heartbeat payloads
- **#2** Search-space validation is incomplete in `launch_run`
- **#5** Broad exception handling hides root causes during heartbeat updates
- **#6** Failed Modal status classification is too coarse
- **#3** `epochs_completed` may be undercounted in W&B metrics reader

### Fix next (important reliability / experiment quality)

- **#9** Tokenization uses process-randomized `hash(...)` (reproducibility risk)
- **#4** W&B auth/setup requirement should be explicit in README (ops reliability)

### Fix later (code quality / architecture / planning alignment)

- **#7** Direct mutation of `state._active_runs` from orchestrator
- **#8** Unused code / unclear parameter usage in `training/modal_app.py`
- **#10** Phase 1 MVP scope mismatch vs current default behavior
- **#11** Phase 1 success criteria are not test-enforced

### Resolved

- **#1** `best_completed_val_loss()` can crash on failed runs
- **#12** W&B history parsing breaks when pandas is unavailable

### Notes

- This ordering intentionally prioritizes **bugs and failure diagnosability** over phase-structure concerns.
- Phase-alignment and test-coverage issues remain logged for visibility, but are intentionally ranked lower than runtime bugs.

---

## 1) `best_completed_val_loss()` can crash on failed runs

**Severity:** High
**Status:** Resolved

If failed runs are moved into `completed_runs` with `best_val_loss: null`, then
`min(losses)` can raise a `TypeError` when `None` is present.

**Why it matters:** Plot generation and heartbeat logic can fail unexpectedly once failed runs accumulate.

**Suggested fix:** Filter to numeric values only before calling `min(...)`.
**Resolution:** Implemented in `state.py` by filtering `best_val_loss` to numeric types before `min(...)`.

---

## 2) Search-space validation is incomplete in `launch_run`

**Severity:** Medium

`launch_run` currently validates parameter names, but does not validate:
- type correctness (e.g. number vs string),
- categorical membership,
- continuous range bounds (`min <= value <= max`).

**Why it matters:** Invalid parameter payloads can be launched and fail remotely.

**Suggested fix:** Add strict validation against the active search-space schema before dispatching to Modal.

---

## 3) `epochs_completed` may be undercounted in W&B metrics reader

**Severity:** Medium

`epochs_completed` is derived from rows filtered on non-null `val_loss`. If validation is logged less frequently than training, epoch progress can be under-reported.

**Why it matters:** Agent decisions may be based on incorrect progress signals.

**Suggested fix:** Compute epoch progress from a more stable signal (e.g. non-null `epoch` from full history), while keeping aligned filtering for best-val-loss calculations.

---

## 4) W&B auth/setup requirement should be explicit in README

**Severity:** Medium (operational)

A common failure mode is launching runs without W&B credentials available in the execution environment.

**Observed case:** Modal runs failed at `wandb.init(...)` with:
`wandb.errors.errors.UsageError: No API key configured. Use wandb login to log in.`
Root cause: the key was available locally, but not inside the Modal container.

**What we did to fix it:**
- Updated `training/modal_app.py` to attach
  `secrets=[modal.Secret.from_name("wandb-secret")]` on `@app.function(...)`.
- This allows Modal containers to access `WANDB_API_KEY` via the `wandb-secret` secret.

**Follow-up required:**
- Ensure the secret exists in Modal:
  - `modal secret create wandb-secret WANDB_API_KEY=...`
- Redeploy after code/config changes:
  - `modal deploy training/modal_app.py`
- Update README auth docs to clearly separate:
  - local auth for orchestrator-side W&B reads, and
  - Modal secret injection for container-side training runs.

---

## 5) Broad exception handling hides root causes during heartbeat updates

**Severity:** Medium

`update_runs_from_modal_and_wandb` swallows errors (especially for running runs) and keeps going without preserving failure diagnostics per run.

**Why it matters:** Debugging run failures becomes difficult, and the agent may reason from stale/incomplete state without clear error context.

**Suggested fix:** Store structured error metadata (timestamp + source + message) in run records; only suppress expected transient errors.

---

## 6) Failed Modal status classification is too coarse

**Severity:** Medium

`check_modal_run_status` maps any non-timeout exception to `"failed"`. Temporary API/network errors are indistinguishable from true training failure.

**Why it matters:** Healthy runs may be prematurely marked failed and removed from active tracking.

**Suggested fix:** Classify retryable vs terminal exceptions; apply retries/backoff before marking failed.

---

## 7) `state._active_runs` is mutated directly from orchestrator

**Severity:** Low (code quality / maintainability)

`orchestrator.py` assigns `state._active_runs = still_active` directly, bypassing `WorkspaceState` public API.

**Why it matters:** Breaks encapsulation and makes future state refactors riskier.

**Suggested fix:** Add a public setter/update helper on `WorkspaceState` (e.g., `replace_active_runs(...)`) and use it.

---

## 8) `training/modal_app.py` has unused code and unclear parameter usage

**Severity:** Low (code quality)

Current training script contains:
- unused `import math`,
- a `gpu` function argument that is not used in function logic.

**Why it matters:** Minor, but increases noise and confusion for maintainers.

**Suggested fix:** Remove unused import/arg or document intentional passthrough semantics.

---

## 9) Tokenization uses Python `hash(...)`, which is process-randomized

**Severity:** Medium

The example tokenizer maps tokens via `hash(word) % vocab_size`. Python hash is randomized between processes by default.

**Why it matters:** Run-to-run token IDs are not stable, making experiments less reproducible and comparisons noisier.

**Suggested fix:** Use a deterministic hashing/tokenization approach.

---

## 10) Phase 1 MVP scope mismatch: default behavior allows parallel launches

**Severity:** Medium (phase-alignment)

Phase 1 goal in the plan is "single run + sequential experiments", but default config (`max_concurrent_runs = 4`) and tool behavior permit immediate multi-run launch.

**Why it matters:** Real behavior can skip intended MVP validation path and jump into Phase 2-style operation before stability is proven.

**Suggested fix:** For Phase 1 validation, default to `max_concurrent_runs = 1` (or add explicit phase gate).

---

## 11) Phase 1 success criteria are not test-enforced

**Severity:** Medium (validation gap)

Plan-level Phase 1 criteria include:
- sensible search-space initialization from model description,
- sequential run iteration,
- at least one search-space update after multiple runs.

Current tests validate tool mechanics and mocked flow, but do not enforce these behavioral outcomes end-to-end.

**Why it matters:** MVP may appear complete while core autonomous optimization behavior is unverified.

**Suggested fix:** Add integration-style acceptance tests aligned to Phase 1 success criteria.

---

## 12) W&B history parsing breaks when pandas is unavailable

**Severity:** High
**Status:** Resolved

When the orchestrator environment lacks `pandas`, W&B history calls return list-like data (or require non-pandas mode), but current code assumes DataFrame-style access such as:
- `history["epoch"]`
- `history["val_loss"]`
- `.dropna()`

This causes runtime errors like:
`list indices must be integers or slices, not str`
and prevents both metric updates and plot generation.

**Why it matters:** The agent sees stale run state (e.g., `epochs_completed = 0`) and makes incorrect decisions even when training is progressing.

**Suggested fix:**
- Add `pandas` as an explicit dependency for orchestrator/runtime consistency.
- Make `tools/wandb_reader.py` and `tools/plotter.py` robust to both DataFrame and list-of-dicts history formats (normalize before processing).
- Avoid silent failure paths so stale metrics are surfaced clearly.
**Resolution:** Implemented:
- Added `pandas` to `pyproject.toml` dependencies.
- Added history normalization in `tools/wandb_reader.py` (`_normalize_history`) and reused it in `tools/plotter.py` so both DataFrame and list-like history work.

---

## 13) Claude 429 rate-limit failures from oversized heartbeat payloads

**Severity:** High
**Status:** Partially addressed

Heartbeats can exceed Anthropic organization input-token-per-minute limits when prompt payloads are large (skills + full JSON state + multiple plot images), especially during chained tool-use calls in the same minute.

**Observed case:** API error:
`rate_limit_error ... exceed ... 10,000 input tokens per minute`

**Why it matters:** A heartbeat can fail after partial tool execution, leaving stale strategy/planning and reducing agent reliability.

**Suggested fix:**
- Reduce heartbeat payload size (summarize/truncate state, include fewer plots, avoid redundant prompt text).
- Add explicit guidance to avoid `launch_run` when `active_runs >= max_concurrent_runs`.
- Add retry/backoff handling for 429 responses.
- Consider lower-frequency or adaptive image inclusion.
**Progress so far:**
- Added `anthropic.Anthropic(max_retries=4)` in `orchestrator.py`.
- Continuation calls now strip image blocks to reduce repeated token load in tool-call loops.
**Remaining gap:** Initial heartbeat requests can still exceed TPM limits with large state + multiple images.

