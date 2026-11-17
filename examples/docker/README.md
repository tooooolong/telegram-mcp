# Docker Deployment Guide for Telegram MCP

This directory provides ready-to-use Docker Compose templates and client configurations for running `telegram-mcp`.

## Architectural Modes

Depending on your client setup and workload, choose one of two deployment architectures:

| Mode | Transport | Best For | Pros | Considerations |
| :--- | :--- | :--- | :--- | :--- |
| **Shared Server** *(Recommended)* | Streamable HTTP (`MCP_TRANSPORT=http`) | Multiple clients, subagent teams, persistent servers | Single long-lived connection, avoids Telegram session rate-limits, shared transcript cache | Publishes on localhost and requires a bearer token |
| **Per-Client Stdio** | Stdio (`MCP_TRANSPORT=stdio`) | Single CLI tool, isolated one-shot execution | Isolated ephemeral container lifecycle | Spawns a container per client session; not recommended for multi-agent suites |

---

## 1. Shared Server Mode (Recommended)

### Quickstart

1. Prepare your `.env` file with credentials:
   ```bash
   install -m 600 .env.example .env
   # Set TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_STRING, and MCP_AUTH_TOKEN
   ```

2. Ensure the transcript cache directory exists and has correct ownership for the non-root `appuser` (UID 1000):
   ```bash
   mkdir -p ./transcript_cache
   chown -R 1000:1000 ./transcript_cache  # On Linux/macOS
   ```

3. Start the shared service:
   ```bash
   docker compose -f examples/docker/docker-compose.shared.yml up -d
   ```

The server binds to `127.0.0.1:8765`.

### Client Configurations

Set `MCP_AUTH_TOKEN` in the client's environment to the same token used by the
server. See the [client commands](../../README.md#transports) for Claude Code and
Codex. Use [per-client stdio](#2-per-client-stdio-mode) for Claude Desktop.

#### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "telegram": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer ${env:MCP_AUTH_TOKEN}"
      }
    }
  }
}
```

---

## 2. Per-Client Stdio Mode

If your client directly manages the Docker process via stdio:

### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "telegram-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--env-file",
        "/absolute/path/to/.env",
        "telegram-mcp:latest"
      ]
    }
  }
}
```

---

## 3. Multi-Account Support

To expose multiple Telegram accounts through Docker, supply distinct session strings in your `.env`:

```env
TELEGRAM_SESSION_STRING_PERSONAL="1BVts..."
TELEGRAM_SESSION_STRING_WORK="1BVts..."
```

Tools accept the `account` parameter (e.g. `send_message(..., account="work")`).

---

## 4. Troubleshooting & Operational Notes

- **Port Binding:** Publish the HTTP port on `127.0.0.1` (`-p 127.0.0.1:8765:8765`). The MCP endpoint requires the shared bearer token.
- **Cache Persistence:** Mounting `/app/data/transcripts` persists SQLite voice-transcription caches across container rebuilds.
- **Permission Denied:** If the container logs show permission errors on `/app/data`, verify that `./transcript_cache` on the host is writable by UID 1000.
