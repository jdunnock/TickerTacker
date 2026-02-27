from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Invalid JSON in {config_path}. Please fix it first.")
            return
    else:
        config = {}

    mcp_servers = config.setdefault("mcpServers", {})

    command = f"cd '{repo_root}' && source .venv/bin/activate && python -m app.mcp_server"

    server_entry = {
        "command": "/bin/zsh",
        "args": ["-lc", command],
        "cwd": str(repo_root),
    }

    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "DATABASE_URL":
                server_entry["env"] = {"DATABASE_URL": value.strip()}
                break

    mcp_servers["tickertacker"] = server_entry

    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("Claude MCP config updated:")
    print(config_path)
    print("\nServer name: tickertacker")
    print("Restart Claude Desktop after this.")


if __name__ == "__main__":
    main()
