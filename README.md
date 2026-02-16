# Overmind Worker

Remote worker node for [Overmind](https://github.com/trianglechoke/overmind) — connects to the central server via WebSocket and executes tool calls locally (bash, file ops, screenshots, GUI).

**No API keys required.** The worker only executes tools; all LLM calls happen on the server.

## Quick Start

```bash
git clone https://github.com/trianglechoke/overmind_worker.git
cd overmind_worker
pip install -r requirements.txt

# Optional: GUI capabilities (screenshot, mouse, keyboard)
# pip install pyautogui pygetwindow

# Run with auto-update
python worker_service.py \
    --server ws://OVERMIND_HOST:8000/ws/worker \
    --machine-id my-machine
```

## Usage

### worker_service.py (recommended)

Auto-updating watchdog — pulls latest code every 60s and restarts the worker if code changed.

```bash
python worker_service.py \
    --server ws://192.168.1.100:8000/ws/worker \
    --machine-id mac-builder \
    --workspace ./projects \
    --pull-interval 30
```

### Direct client (no auto-update)

```bash
PYTHONPATH=src python -m overmind.workers.client \
    --server ws://192.168.1.100:8000/ws/worker \
    --machine-id mac-builder \
    --workspace ./projects
```

## Requirements

- Python 3.12+
- `websockets` (WebSocket connection)
- `httpx` (transitive dependency)
- Optional: `pyautogui`, `pygetwindow` (for GUI/screenshot capabilities)

## How It Works

1. Worker connects to Overmind server via WebSocket
2. Registers with machine ID, hostname, and detected capabilities
3. Receives tool call requests (bash, read_file, write_file, etc.)
4. Executes them locally and sends results back
5. Auto-reconnects on connection loss (3s retry)
