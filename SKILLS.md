# SKILLS.md

Task router for agents in this repo.

## 1) Data exploration
Use when you need topic discovery, coverage checks, data sanity, or backfill validation.
- Guide: `skills/allora-data-exploration/SKILL.md`
- Output: clear data/topic readiness summary

## 2) Model building
Use when you need feature work, training, evaluation, or quality improvement.
- Guide: `skills/allora-model-builder/SKILL.md`
- Output: artifact + metrics + recommendation

## 3) Worker operations
Use when you need deployment, local worker lifecycle, or submission monitoring.
- Guide: `skills/allora-worker-manager/SKILL.md`
- Output: running worker + monitor/dashboard status

---

## Routing rules
- “What topic/data should I use?” → Data exploration
- “Improve prediction quality” → Model building
- “Go live / deploy / monitor” → Worker operations

For release tasks: run in order **Data → Model → Worker ops**.
