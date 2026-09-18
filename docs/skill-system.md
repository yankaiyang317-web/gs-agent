# Skill status and candidate design

## Current status

gs-agent does not ship a formal `SKILL.md` package in the current version.
This is deliberate. The project already has reusable workflows, but they are
currently expressed through `AGENTS.md`, task templates, MCP tool contracts,
and runtime logic.

The absence of a formal Skill does not mean that coverage exploration or target
finding is unimplemented. It means those workflows have not yet been packaged
with automatic Skill discovery and triggering.

## What a Skill would add

A formal Skill would package:

- Trigger conditions.
- Required inputs.
- A reusable step-by-step workflow.
- Tool-selection guidance.
- Safety and recovery rules.
- Completion criteria.
- References loaded only when the Skill is active.

Packaging may also change behavior by affecting automatic triggering,
instruction precedence, context size, and which references are loaded. Those
effects should be evaluated before enabling a Skill.

## Existing reusable workflows

### Coverage-first exploration

Existing sources:

- Coverage rules in `AGENTS.md`.
- Coverage task template in `docs/agent/task_templates.md`.
- Candidate and checkpoint tools exposed by MCP.
- Coverage topology in `gs_mcp/campaign.py`.

Possible future Skill trigger:

> The user asks to freely explore a 3DGS scene for broad coverage.

Expected completion:

> The configured Campaign clip target is complete, or an explicit navigation
> budget is exhausted and the remaining routes are reported.

### Target finding

Existing sources:

- Navigation and target-approach rules in `AGENTS.md`.
- Target-finding task template.
- RGB/depth observations and `gs_approach_target`.
- Collision-aware movement and navigation guards.

Possible future Skill trigger:

> The user asks to find a named object, place, entrance, or semantic region.

Expected completion:

> The target is confirmed from visual evidence and reached from a safe nearby
> viewpoint, or the searched routes and failure evidence are reported.

### Reconstruction-quality inspection

This is another candidate, but it should remain separate from coverage because
its finding-recording behavior and completion condition are different.

## Why Skill packaging is deferred

Before introducing a formal Skill, the project should compare the existing
Prompt-only baseline against Skill-enabled runs:

- Correct task-family selection.
- User-instruction precedence.
- Tool-call sequence and error recovery.
- Coverage and repetition metrics.
- Target-finding success rate.
- Context/token overhead.
- Behavior when several candidate Skills could match.

The initial Skill should be added only if it improves reuse or reliability
without silently changing established behavior.

## Proposed package shape

A future package could use:

```text
skills/
|-- gs-coverage-exploration/
|   |-- SKILL.md
|   `-- references/
|       |-- navigation.md
|       `-- campaign.md
`-- gs-target-finding/
    |-- SKILL.md
    `-- references/
        |-- target-approach.md
        `-- recovery.md
```

Each `SKILL.md` should identify:

- When it should and should not trigger.
- Inputs such as scene and clip target.
- The first required configuration call.
- The supported MCP tools.
- Runtime feedback that must be treated as hard evidence.
- Completion and failure reporting.

## Decision record

Current decision: keep reusable behavior in Prompt documentation and runtime
code while the Harness boundary is clarified. Revisit Skill packaging after
repeatable evaluation tasks exist for both coverage and target finding.
