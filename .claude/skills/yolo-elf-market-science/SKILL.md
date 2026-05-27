---
name: yolo-elf-market-science
description: Build scientific product, market, and experiment analysis around YOLO Elf. Use when comparing target users, positioning, demo strategy, privacy tradeoffs, model/runtime benchmarks, pricing hypotheses, adoption metrics, or evidence-based recommendations. Browse current sources when making present-day market claims.
---

# YOLO Elf Market Science

## Focus

Turn product and performance evidence into testable decisions. This repo has benchmark and status signals, but no real acquisition, retention, or revenue data yet; label assumptions clearly and separate market research from measured runtime facts.

## File Map

- Product surfaces: `README.md`, `static/phone.html`, `static/viewer.html`, `static/demo-frame.svg`.
- Performance evidence: `scripts/bench_detector.py`, `scripts/bench.ps1`, `app/stream_state.py`, `app/detector.py`.
- Tunable parameters: `app/config.py`, `TUNING.md`, `OPTIMIZATION_PLAN.md`.
- Privacy and demo constraints: `README.md`, `app/remote_storage.py`, `scripts/build-static.mjs`.

## Analysis Modes

- Market framing: define user segment, job-to-be-done, alternatives, differentiation, and adoption barrier.
- Experiment design: define hypothesis, metric, sample, instrumentation, success threshold, and rollback decision.
- Runtime science: compare model/device/img size/JPEG quality/FPS using benchmark outputs.
- Demo strategy: explain what the static GitHub Pages demo can prove and what live capture must prove separately.

## Evidence Rules

- Use local benchmark/status data for performance claims.
- Browse or cite current sources for market-size, competitor, pricing, or regulation claims.
- Do not invent funnel metrics; propose instrumentation when data is absent.
- Treat camera frames and remote uploads as privacy-sensitive in market recommendations.
- Tie every recommendation to a measurable next step.

## Workflow

1. State the decision being supported.
2. Inventory available evidence from repo files, benchmarks, docs, or current sources.
3. Separate facts, assumptions, and unknowns.
4. Recommend the smallest experiment or product change that can reduce uncertainty.
5. Include implementation hooks when instrumentation or product copy changes are needed.

## Validation

For runtime experiments, run or request benchmark outputs. For market claims, cite current sources. For docs/product changes, run `npm.cmd run test` and `npm.cmd run build` when touched files affect the static demo.
