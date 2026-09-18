# Agent Harness

## Definition

The Harness is the execution framework between an external vision-capable Agent
and a 3DGS scene. It turns model decisions into validated environment actions,
maintains task state, applies hard safety constraints, and records reproducible
evidence.

The Agent/model loop itself is not implemented by gs-agent. It is hosted by an
MCP client such as Codex. gs-agent implements the environment side of that
loop.

## Current boundary

The Harness is already functional, but its responsibilities are distributed
across existing modules instead of being exposed through one public
`GSAgentHarness` class.

| Responsibility | Current implementation |
| --- | --- |
| MCP transport and tool registration | `gs_mcp/server.py` |
| Persistent scene and Campaign lifecycle | `gs_mcp/runtime.py` |
| Tool execution, validation, and navigation guards | `gs_mcp/tools.py` |
| Rendering, camera motion, geometry, and collision | `gs_env/` |
| Coverage topology and exploration memory | `gs_mcp/campaign.py` |
| Campaign persistence and frame ownership | `gs_mcp/campaign_base.py` |
| Observation and trajectory recording | `gs_mcp/debug_writer.py` |
| Video export and completion handling | `gs_mcp/video.py` |
| Contract and regression validation | `tests/` |

## Control flow

```text
User task
    |
    v
Vision-capable Agent / MCP client
    |
    v
MCP tool schema and server
    |
    v
SceneRuntime + EnvironmentTools
    |
    +--> GSEnvironment --> renderer / camera / collision
    |
    +--> ExplorationCampaign --> checkpoints / routes / coverage
    |
    +--> DebugObservationWriter --> frames / trajectory / video
    |
    v
Structured tool result returned to the Agent
```

## Lifecycle

### Persistent HTTP mode

1. `gs_mcp.runtime_server` starts with a scenes directory and output roots.
2. The MCP service registers a stable tool schema before a scene is loaded.
3. `gs_configure_campaign` validates the scene, objective, profile, and clip
   target.
4. `SceneRuntime` resolves the manifest and loads renderer and collision
   assets.
5. A new isolated Campaign directory is created.
6. `EnvironmentToolsProxy` binds the stable MCP tools to the loaded scene.
7. Agent tool calls update the environment, Campaign, and recordings.
8. Completion or shutdown finalizes pending video output and releases scene
   resources.

An incomplete Campaign cannot be replaced by another scene or runtime profile.

### Local stdio mode

1. The MCP client starts `gs_mcp.server` with one scene manifest.
2. The scene is loaded during process startup.
3. An optional Campaign root enables exploration state and recording.
4. The client configures the clip target using the shorter already-loaded-scene
   form of `gs_configure_campaign`.
5. Process lifetime is owned by the MCP client.

## Observation and action contract

Observations provide:

- RGB image.
- Depth preview.
- Alpha-aware depth validity information.
- Camera-to-world pose and intrinsics.

Actions include:

- Pose inspection and debug placement.
- Collision-aware translation and rotation.
- Visible-target approach.
- Route and viewpoint candidate reporting.
- Exploration status queries.
- Checkpoint restoration for supported recovery cases.

The Harness returns structured action results rather than hiding partial
movement or collision recovery.

## Hard runtime constraints

The Harness, not the Prompt, enforces:

- Scene manifest resolution and validation.
- Single-active-Campaign ownership.
- Tool argument validation.
- Swept collision checks.
- Navigation recovery feedback.
- Alpha/depth validity for target approach.
- Checkpoint creation and topology ownership.
- Restore eligibility.
- Clip target and completion accounting.

This keeps core safety and state consistent even when Prompt wording changes.

## State and reproducibility

Each Campaign owns an isolated run directory containing:

- `campaign.json` for versioned task state and parameters.
- `trajectory.jsonl` for action and pose history.
- Debug observations and frame metadata.
- Exported video clips.

Scene-specific inputs are captured by versioned manifests. Large PLY and GLB
assets remain external to Git but retain stable repository-relative paths.

## Current limitations

- No built-in model inference loop; the MCP client supplies the Agent.
- Semantic routes and targets are proposed by the visual Agent.
- The Harness implementation is distributed across modules.
- Only one active scene/Campaign is owned by one persistent runtime.
- Batch scheduling and concurrent Campaign ownership are not implemented.
- Real-scene GPU validation remains separate from CPU-focused tests.

## Proposed explicit Harness API

A future refactor can add `gs_mcp/harness.py` without changing the MCP tool
contract:

```python
class GSAgentHarness:
    def configure_campaign(self, ...): ...
    def observe(self): ...
    def move(self, ...): ...
    def rotate(self, ...): ...
    def approach_target(self, ...): ...
    def report_candidates(self, ...): ...
    def get_status(self): ...
    def restore_checkpoint(self, ...): ...
    def finalize(self): ...
```

The class would compose the existing runtime, environment tools, Campaign, and
writer. `server.py` would remain a thin MCP adapter.

## Refactor sequence

1. Freeze the current MCP tool schemas with contract tests.
2. Add a typed Harness configuration object.
3. Introduce `GSAgentHarness` as a facade over existing modules.
4. Route MCP tools through the facade without moving implementation code.
5. Add `tests/test_harness.py` for lifecycle and error behavior.
6. Move internal ownership only after behavior matches the baseline.

The first refactor should not change Prompt behavior, collision behavior,
Campaign persistence, or output layout.

## Harness test plan

The Harness boundary should verify:

- An unloaded HTTP runtime rejects scene actions.
- A valid manifest creates one isolated Campaign.
- Invalid scene names and asset paths fail clearly.
- An incomplete Campaign blocks scene replacement.
- Movement results preserve collision and partial-progress evidence.
- Direct debug pose placement does not create traversal topology.
- Checkpoint restore follows Campaign rules.
- Completion finalizes the configured number of clips.
- Repeated runs do not share state.

Existing tests already cover many individual behaviors. A dedicated Harness
test should combine them around the public lifecycle boundary.
