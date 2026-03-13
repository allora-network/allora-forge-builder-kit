# SKILLS.md

Task router for agents working in this repository.

## How to use this file

1. Match the user request to one skill below.
2. Read that skill's `SKILL.md`.
3. Execute only the required steps.
4. Keep outputs concise and reproducible.

---

## Available skills

### 1) Model building
**Use for:** feature engineering, target creation, training loops, evaluation, experiment sweeps.

- Path: `skills/allora-model-builder/SKILL.md`
- Typical inputs: topic id, horizon, interval, feature set.
- Typical outputs: metrics report, saved artifact, recommended config.

### 2) Data exploration
**Use for:** topic discovery, data sanity checks, backfill inspection, quick quality checks.

- Path: `skills/allora-data-exploration/SKILL.md`
- Typical outputs: available topics, data coverage/gaps, exploratory summaries.

### 3) Worker operations
**Use for:** deployment, local worker lifecycle, monitoring, dashboard workflow.

- Path: `skills/allora-worker-manager/SKILL.md`
- Typical outputs: deployed worker state, monitor summary, runtime diagnostics.

---

## Routing rules (important)

- If request is ambiguous, default to **data exploration** first.
- If user asks to "go live" or "deploy", use **worker operations**.
- If user asks to improve metrics/model quality, use **model building**.
- Multi-step projects can chain skills, but keep each step explicit.

---

## Release-mode expectation

For release prep tasks, combine all three skills in this order:
1. Data exploration (confirm topic/data assumptions)
2. Model building (train/evaluate/export)
3. Worker operations (deploy/monitor)

Then update README/AGENTS docs to reflect the validated flow.
