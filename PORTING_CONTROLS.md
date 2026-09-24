# Adding the Reflexion / Partial-Reroll / Full-Reroll controls to CRAFT-MD

**Who this is for:** an agent working in `behavioral-interventions-craft-md` (this repo).
**Reference implementation:** `HanavM/tau-bench` (redirects to `behavioral-interventions-tau-bench`), `main` @ `1fbf4cf`. All of this exists and runs there — port it, don't reinvent it. The relevant files are `run_intervenor_sweep.py`, `tau_bench/reflexion.py`, `tau_bench/partial_reroll.py`, `run_baseline_n_times.py`, `scripts/`.

**Goal.** For each CRAFT-MD worker model, produce the same row the tau-bench table has:

| row | what it measures | status here |
|---|---|---|
| `None` | baseline pass rate | exists |
| `<intervenor>` Bo-N | best of N interventions at the identified failure point | exists (but see §0.4) |
| `<intervenor>` single | execute only the intervenor's **first** proposal | derivable offline, no new rollouts |
| `None (Reflexion)` | up to N from-scratch retries, each seeded by a self-written reflection | **missing entirely** |
| `None (Partial RR)` | N rerolls from the same truncation point with **no text inserted** | **missing** (90% of the primitive exists) |
| `None (Full RR)` | extra complete runs at new seeds, best-of-N with the baseline | **missing** (skeleton only, see §3c) |

The controls are what make the intervention number meaningful: partial RR isolates the *text* from resampling at the same point; full RR isolates it from simply trying again. In tau-bench the controls beat the intervenors on several worker models — expect that to be a real possible outcome here, not a bug.

---

## 0. Facts about this repo that change how you port

1. **`reward` is a `bool`**, not tau-bench's `1.0`/`0.0`. Every ported `r["reward"] == 1.0` becomes truthiness. Normalise once at the boundary.
2. **The trajectory is not `record["traj"]`.** It is `record["trial_0"]["multiturn_conversation"]` — a list of `{"role","content"}`. Index 0 is the system prompt, index 1 a fixed greeting, and **doctor replies are on odd indices** (the intervenor prompt depends on this, `src/prompts.py:211`). Write one accessor and use it everywhere.
3. **Scoring is an LLM judge**, not an env reward: `src/craftmd.py:371-389` (duplicated at `:752`, `:967`, `:1339`) asks `gpt-4o-mini` at `temperature=0.0` whether the last doctor message matches ground truth. The judge model is **hardcoded in 4 places**. Every control must score through the same judge or its number is not comparable. Factor it into one function while you are there.
4. **The live intervention path uses `N=3, temperature=0.1`** (`src/craftmd.py:1128`), not N=5. tau-bench's table is Bo-5. Decide deliberately: either re-run interventions at N=5 or label the CRAFT-MD column Bo-3. Do not silently compare Bo-3 against Bo-5 controls.
5. **There is no max-turn cap.** The dialogue loop is `while True:` and stops only when the doctor's reply has no `"?"` or contains `'Final Diagnosis'` (`src/craftmd.py:943`). The 40-turn limit exists only as prose in the prompt (`src/prompts.py:47`). Add a real cap before you multiply rollouts by N.
6. **Best-of-N is offline.** Nothing in this repo aggregates it: the loop writes one record per intervention tagged `intervened_first_or_last = 0..N-1` (`src/craftmd.py:1243`), and `did_improve` is computed but unused. Aggregation is `max(reward)` grouped by `case_id` — the same convention tau-bench's reporting uses.

## 1. Fix these first — they corrupt results silently

Each control multiplies rollouts by N, so these get worse, not better.

- **Concurrent read-modify-write on one file.** The append pattern (`open read → json.load → append → open write`) is repeated ~8× verbatim (`src/craftmd.py:468, 511, 567, 821, 1036, 1098, 1141, 1163, 1408`) and runs inside `multiprocessing.Pool(10)` (`parallel_craftmd_gpt.py:66,78`). Workers finishing together lose records. This already shows up as stray `{}`/`[]` elements in committed results. Fix with a lock, or one file per case merged afterwards (what tau-bench does in `fold_topups`).
- **Dead resume.** `src/craftmd.py:868-872` loads `transcript.json` and tests `f'trial_{j}' in stats` — but the file is a *list* of case records, so the test is always False, and `stats` is reset to `{}` at `:878` anyway. A rerun re-runs everything and appends duplicates. Every control depends on real resume: skip a case whose record exists and is complete.
- **No retry/backoff and no timeout.** `chat_text` (`src/craftmd.py:58`) is a bare `client.chat.completions.create(...)` — a 429 raises, and there is no timeout, so a wedged socket hangs the worker forever. (The retry code in `src/models.py` is legacy and unused.) In tau-bench, missing timeouts wedged jobs for hours; see `tau_bench/llm_utils.py:completion_with_backoff`.
- **Infinite loop on a `None` response.** In the trial loop, `j` only increments inside `if flag == 0:` (`src/craftmd.py:1033`), so a failed call spins forever; `success_pre_intervention` is also referenced unbound at `:1056`. Fix before enabling multi-trial anything.
- **`craftmd_gpt_baseline` has no exception handling** (the `try` is commented out at `:856`), so one API error kills that case's worker silently.

## 2. Give the runner a CLI

`parallel_craftmd_gpt.py` is configured by editing the file: models at `:19-21`, a `run_baseline` boolean at `:30`, a hardcoded `baseline_folder` at `:46`, `case_min/case_max` at `:76-77`. A sweep driver cannot drive that. Add `argparse`:

```
--mode {baseline,intervention,reflexion,partial_rr,full_rr}
--doctor-model / --patient-model / --intervenor-model / --judge-model
--baseline-path <dir>        # required by every non-baseline mode
--best-of-N 5
--num-trials 5               # full_rr only
--temperature 1.0
--case-ids ...               # top-ups: run ONLY these cases
--categories "Neurology,Infectious Diseases"
--log-dir ./results
--max-concurrency 10
```

`--case-ids` is not optional — the whole resumable design depends on re-running exactly the gap.

## 3. Implement the three controls

### 3a. Reflexion — port `tau_bench/reflexion.py`

Nothing exists here (only an incidental use of the word in `src/prompts.py:244`). For each case that **failed at baseline**, up to `best_of_N` sequential attempts: ask a reflection model (tau-bench used `gpt-4o`, deliberately separate from the doctor model) to write why the last attempt failed, then run a **fresh dialogue from scratch** with that reflection prepended to the doctor's system prompt. Stop as soon as an attempt passes.

- Output `<baseline>/reflexion-by-<writer>-<id>/reflexion-transcripts.json`, **one row per attempt**, with `attempt_number` and `reflection_text`.
- Terminal when: passed at baseline, succeeded on an attempt, or used `best_of_N` attempts. **Count errored attempts toward the budget** — tau-bench's driver didn't, and it asked forever for attempts the script refused to run.

### 3b. Partial reroll — port `tau_bench/partial_reroll.py`

The primitive already exists: `add_intervention` (`src/craftmd.py:1180`, duplicated at `:584`) truncates and injects:

```python
new_trajectory = trajectory[:min(idx_intervention + 1, len(trajectory) - 1)]
new_trajectory.append({"role": "system", "content": "[*INTERVENTION*: " + intervention_text + "]"})
```

**Partial reroll is that, minus the append.** Two details that must match the intervention run exactly or the control is invalid:
- Truncate at `intervention["id"]` (the *insertion* index), **not** `failure_id` — they are different fields.
- Reuse the same `run_intervention` call to pick the point, so the marker is chosen identically. Record `intervened_message = "(no text inserted - partial reroll)"` and keep `intervened_index`.
- Temperature must be non-zero (`chat_text` defaults to 1.0) or all N rerolls are identical.

⚠️ **Inherited bug to fix first.** The continuation code builds the patient history from the doctor's (`src/craftmd.py:1283-1286`), so the patient model sees roles inverted relative to the baseline loop. The role-swap fix exists but is commented out in `craftmd_gpt` (`:690-698`) and absent from `craftmd_gpt_intervention`. Any reroll reusing this continuation inherits it — rebuild `conversation_history_patient` by swapping `user`↔`assistant` over the truncated prefix.

### 3c. Full reroll — skeleton only, more work than it looks

`num_runs` exists (`src/craftmd.py:853`, default 1) and drives `while j < num_runs:` writing `trial_j`, and sampling is at temperature 1.0, so trials would genuinely differ. **But** it is not usable as-is: the driver never passes `num_runs`; resume is dead (§1); `j` can loop forever (§1); and **all downstream intervention code hardcodes `trial_0`** (`:71, 77, 92, 1087, 1262, 1277, 1280`) with `craftmd_gpt_intervention` hardcoding `while k < 1` (`:1266`).

Two options:
- **Fix the trial loop** and teach the downstream code to select a trial. Cheaper on compute, more edits.
- **Port `run_baseline_n_times`** (`tau_bench/run.py:177`): N-1 extra whole runs, each in its own seed folder, resumable per seed. Fewer edits to existing code, easier to top up when a run dies halfway. Note there is no `seed=` passed to any OpenAI call here, so "seeds" are only folder labels — variance comes from temperature 1.0. Say so in the caption.

## 4. The sweep driver

`run_intervenor_sweep.py` (587 lines) is worth copying wholesale — it is benchmark-agnostic apart from paths and record accessors. It provides a cell table in `baselines.json`, jobs that re-derive what is on disk and run only the gap, bounded retries, a stall watchdog (kills a child with <5s CPU in 15 min), a low-disk pause, and atomic writes.

Adapt: task count → case count per category; `r.get("traj")` → the `trial_0.multiturn_conversation` accessor; `reward == 1.0` → truthiness; folder pattern `intervened-by_<iv>_<ts>` → this repo's `intervened-by-<iv>-<id>`.

**Four bugs it already contains fixes for — do not re-introduce them.** All four corrupt results silently:

1. **Canonical folder chosen by mtime.** A top-up killed before its merge leaves a newer, smaller folder; "newest" made a finished cell look ~90% missing.
2. **Retries of "no intervention was done" discarded.** The merge only added cases absent from the canonical file, so a retry that finally produced an intervention was dropped every time. This one inverted a published conclusion — an intervenor looked like it "fails to propose" when the cause was the merge.
3. **Partial best-of-N counted as complete.** One row per candidate means a kill mid-case leaves 3 of 5. Require all N (or a trusted end-of-run marker).
4. **Errored attempts not counted toward a task's budget**, so coverage never closes.

Also note two data-shape traps when computing coverage here: records with no `trial_*` key are written for already-passing cases and for intervenor failure (`:1090-1097`, `:1133-1140`) — and **the intervenor-failure record sets `reward=True`**, which looks like a bug; treat it as the pre-intervention reward, and confirm with the repo owner before reporting any number that depends on it.

## 5. Acceptance checks

1. `--mode baseline --num-trials 1` reproduces an existing `results/*/transcript.json` pass rate for the same models and cases.
2. Kill any control mid-run and re-run: it resumes, adds no duplicate rows, re-runs no completed case.
3. Partial RR truncates at the *same* index the intervention run used for the same case — assert on a sample.
4. For one worker model, all six rows exist and every cell reports full coverage (attempted == baseline failures).
5. Port `scripts/check_table_numbers.py`: recompute every cell from the transcripts and diff against the LaTeX. Do not hand-transcribe numbers into the paper.

## 6. Reporting conventions to match

- Denominator is the **full case count** for the category, not the number of usable transcripts; a case with no usable trajectory counts as a failure.
- A case counts as solved only if a **non-errored** attempt passed.
- "single" rows = the intervenor's **first** proposal (candidate index 0 in `intervened_first_or_last`), derived offline from the same Bo-N data.
- Report coverage with every cell. In tau-bench, two partial cells produced conclusions that reversed once they completed — don't interpret a cell below full coverage.
