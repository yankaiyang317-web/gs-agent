# Prompt system

## Purpose

Prompts define how a vision-capable Agent should use the gs-agent tools. They
select an exploration strategy, describe evidence requirements, and tell the
Agent how to react to runtime feedback. Prompts do not replace collision
checking, argument validation, Campaign state, or other programmatic safety
rules.

## Current Prompt sources

| Source | Role |
| --- | --- |
| Root `AGENTS.md` | Repository-scoped operating policy for navigation and inspection tasks. |
| `docs/agent/task_templates.md` | Optional task prompts for coverage exploration, target finding, reconstruction-quality inspection, and direct movement. |
| MCP tool names, schemas, and docstrings | Action-level affordances, required arguments, and result semantics. |
| Current user request | Scene, objective, target, clip count, and task-specific instructions. |

The project does not embed one monolithic system prompt in Python. Behavior is
assembled from these layers so that general rules, task selection, and tool
contracts can evolve independently.

## Instruction precedence

The expected order is:

1. Platform and MCP client safety requirements.
2. The user's explicit current-task instruction.
3. Repository defaults in `AGENTS.md`.
4. Optional task templates.
5. Tool descriptions and runtime feedback.

An optional template must not override an explicit user instruction. For
example, a target-finding request should not silently become free coverage
exploration because a coverage template is available.

## Main task families

### Coverage-first exploration

Goal: obtain broad spatial coverage rather than inspect one object or maximize
the number of visual defects.

Key Prompt rules:

- Configure the exact scene and clip target before movement.
- Observe before consequential movement and after meaningful view changes.
- Report a small number of visually supported macro routes and viewpoints.
- Treat ordinary corridor or road poses as transit checkpoints.
- Treat junctions, open areas, stairs, and collision recovery as decisions.
- Use Campaign status for local continuation, repetition, and backtracking.
- Continue across video boundaries until the configured target is complete.
- Do not call reconstruction-finding tools for a coverage objective.

### Target finding

Goal: locate a user-requested semantic target and move to a safe nearby view.

Key Prompt rules:

- Search using RGB and depth evidence.
- Explore plausible routes until the target is visually supported.
- Prefer one identifiable target pixel for depth-consistent approach.
- Use a bounding box only for one tight, approximately single-depth surface.
- Stop at a safe distance and report blocked or rejected routes.

### Reconstruction-quality inspection

Goal: find spatially distinct low-quality or incomplete reconstructed regions.

Key Prompt rules:

- Treat visual defects as targets rather than automatic navigation barriers.
- Observe them from safe nearby viewpoints.
- Record spatially distinct findings and continue after the first result.
- Mark a direction as a possible scene boundary only with repeated visual or
  collision evidence.

### Direct movement

Goal: follow explicit user movement instructions subject to hard runtime safety
feedback. Report partial progress when collision prevents the full request.

## Prompt versus runtime enforcement

Prompt-guided behavior includes route selection, semantic interpretation, and
deciding when a view provides useful evidence.

Programmatically enforced behavior includes:

- Tool argument validation.
- Scene and Campaign lifecycle checks.
- Collision sampling and navigation guards.
- Depth/alpha validity for target approach.
- Checkpoint ownership and restore eligibility.
- Clip and Campaign completion accounting.

This separation is intentional. Natural-language guidance may be revised
without weakening hard environment constraints.

## Prompt inputs and outputs

Typical inputs:

- Scene name and objective.
- Exact video clip target.
- Current RGB/depth observation.
- Pose and fixed world-up axis.
- Collision/navigation feedback.
- Current checkpoint, route, and coverage status.

Typical outputs:

- One MCP tool call.
- A small set of route or viewpoint candidates.
- A concise final report with success, evidence, and blocked routes.

## Maintenance and versioning

Prompt changes are source changes and should be reviewed like code:

1. Update `AGENTS.md` or the relevant task template.
2. Update this document if precedence or task behavior changes.
3. Add or update tests for any tool contract relied upon by the Prompt.
4. Run controlled tasks from more than one scene or initial viewpoint.
5. Record user-visible behavior changes in Git history.

Useful evaluation questions include:

- Did the Agent choose the requested task family?
- Did explicit user instructions remain higher priority than defaults?
- Did the Agent respond correctly to collision and recovery feedback?
- Did coverage prompts avoid repeated rotations and redundant routes?
- Did target-finding prompts use depth-consistent approach targets?

## Current limitations

The visual Agent still proposes semantic routes and targets. The runtime does
not infer free-space frontiers or semantic objects directly from pixels.
Prompt quality therefore affects exploration decisions, while collision and
Campaign rules remain the authoritative execution boundary.
