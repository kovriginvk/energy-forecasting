# Project Description

Python machine learning project to forecast electricity carbon intensity in the
Netherlands for the next 7 days at hourly resolution (168 values), using renewable
energy production forecasts as inputs.

The project covers data preparation, model development and evaluation, and a
reproducible workflow with a README and a short report. See
`docs/initial_assignment.md` for requirements and `docs/dev_log.md` for decisions.

## Target-only forecasting

Target-only models can forecast the full next week by shifting the alignment of
historical feature rows and prediction times. The dataset is indexed by feature
time `t`, with `y(t) = target(t + 168 hours)` created by `target.shift(-168)`.
Each model predicts one value per input row. At forecast origin `T`, the 168
feature rows from `T - 167 hours` through `T` produce predictions for `T + 1 hour`
through `T + 168 hours`. A fixed per-row lead of 168 hours therefore supports a
168-value weekly forecast when applied to this historical block.

Use only target observations available at each feature row's time; calendar
features describe its predicted hour and are known in advance. SMA receives
preceding history to compute seasonal averages. The model must be fitted only
on labels available at forecast origin `T`. No future target observations or
recursive prediction feedback are needed to construct the weekly forecast.
The current experiment runner scores individual rows with a single fit; it does
not simulate successive weekly forecast origins or weekly refitting.

# Development Guidelines

- Write all project files in English, including documentation, notebook Markdown,
  code comments, and user-facing text.
- Keep `docs/dev_log.md` in chronological decision order (oldest first), including
  within the same day. Append new entries at the end. Keep each entry to one short
  paragraph or a few bullets covering the decision, reason, and key constraints.
  Preserve historical decisions; record later changes in new entries. Put usage
  instructions in README and implementation details in code instead of the log.

# Behavioral Guidelines

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.
