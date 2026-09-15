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
  "/absolute/path/to/gs-agent/scenes/guju.json",
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

```bash
python -m gs_mcp.runtime_server \
  --scenes-root ./scenes \
  --runs-root ./outputs/exploration_runs \
  --demo-runs-root ./outputs/demo_runs \
  --device cuda \
  --host 127.0.0.1 \
  --port 18913
```

Configure the client:

```toml
[mcp_servers.gs_agent]
url = "http://127.0.0.1:18913/mcp"
startup_timeout_sec = 300
```

A copy-ready example is available at `.codex/config.http.example.toml`.

## Remote HTTP server

Network routing is separate from MCP configuration. Prefer keeping the service
on loopback and forwarding it over an authenticated SSH connection:

```bash
ssh -N -L 18913:127.0.0.1:18913 user@render-host
```

The client still connects to `http://127.0.0.1:18913/mcp`. Substitute any
authorized server name.

## Start a task

The persistent HTTP runtime starts without a scene. First call:

```text
gs_configure_campaign(
    scene="guju",
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
