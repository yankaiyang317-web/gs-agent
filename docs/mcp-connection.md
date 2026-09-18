# Connect an MCP client

gs-agent uses the Model Context Protocol and supports local stdio and persistent
Streamable HTTP transports. The MCP client may be Codex or any compatible
client.

## Local stdio

Use stdio when the MCP client and gs-agent run on the same machine and one
selected scene should be loaded by the process.

```toml
[mcp_servers.gs_agent]
command = "python"
args = [
  "-m",
  "gs_mcp.server",
  "--scene-manifest",
  "/absolute/path/to/gs-agent/scenes/my_scene.json",
  "--exploration-campaign",
  "/absolute/path/to/gs-agent/outputs/exploration_runs",
]
startup_timeout_sec = 300
```

Start the MCP client from an environment where gs-agent and its rendering
dependencies are installed. A copy-ready example is available at
`.codex/config.stdio.example.toml`.

The stdio process already owns one selected scene. Configure its Campaign with:

```text
gs_configure_campaign(video_clips=1, objective="coverage")
```

## Streamable HTTP

Use HTTP when gs-agent should remain running independently of the MCP client,
when several client sessions reconnect to one runtime, or when rendering runs
on another host.

Start the server:

The port is configurable. `18913` is the application default, not an MCP requirement. Replace `<PORT>` below with any available port and use the same value in the client URL.

```bash
python -m gs_mcp.runtime_server \
  --scenes-root ./scenes \
  --runs-root ./outputs/exploration_runs \
  --demo-runs-root ./outputs/demo_runs \
  --device cuda \
  --host 127.0.0.1 \
  --port <PORT>
```

Configure the client:

```toml
[mcp_servers.gs_agent]
url = "http://127.0.0.1:<PORT>/mcp"
startup_timeout_sec = 300
```

A copy-ready example is available at `.codex/config.http.example.toml`.

## Remote HTTP server

Network routing is separate from MCP configuration. Prefer keeping the service
on loopback and forwarding it over an authenticated SSH connection:

```bash
ssh -N -L <LOCAL_PORT>:127.0.0.1:<SERVER_PORT> user@render-host
```

The client connects to `http://127.0.0.1:<LOCAL_PORT>/mcp`. The local forwarding port may differ from the server port; use the local port in the MCP URL.

## Start a task

The persistent HTTP runtime starts without a scene. First call:

```text
gs_configure_campaign(
    scene="my_scene",
    video_clips=1,
    objective="coverage",
    profile="development",
)
```

The runtime then exposes observation, navigation, target approach, campaign
status, route reporting, and checkpoint tools through the same MCP connection.

## Deployment-specific configuration

A deployment may use systemd, Docker, a process supervisor, or a manually
started Python environment. These choices do not change the MCP tool contract.
