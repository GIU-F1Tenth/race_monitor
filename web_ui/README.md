# Celeritas Race Control — Web UI

Live race control dashboard for the `race_monitor` ROS2 node.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python ≥ 3.10 | system |
| Node.js ≥ 18 | `curl -fsSL https://deb.nodesource.com/setup_20.x \| sudo -E bash - && sudo apt-get install -y nodejs` |
| ROS2 Humble | sourced in shell |

---

## Running

**Terminal 1 — backend** (ROS2 must be sourced):

```bash
cd race_monitor/web_ui/backend
pip install -r requirements.txt          # first time only
python3 -m uvicorn main:app --host 0.0.0.0 --port 8082
```

**Terminal 2 — frontend**:

```bash
cd race_monitor/web_ui/frontend
npm install                              # first time only
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Adding the team logo

1. Drop your image into `frontend/public/` — e.g. `frontend/public/logo.png`
2. Open `frontend/src/pages/ControlPanel.tsx`
3. Find the `TeamLogo` function and replace the `<svg>` with:

```tsx
function TeamLogo() {
  return (
    <img
      src="/logo.png"
      alt="Celeritas"
      style={{ width: 42, height: 42, objectFit: 'contain' }}
    />
  );
}
```

---

## Control buttons

| Button | Key | Action |
|--------|-----|--------|
| PAUSE | `P` | Pause race timing |
| RESUME | `R` | Resume after pause |
| RESET LAP | `T` | Discard current lap timer |
| IGNORE / RESET | `I` / `B` | Full reset — no data saved |
| END + SAVE | `F` | End race, save all completed laps |

---

## Odom topic

By default the backend listens on `/odom`, `/ego_racecar/odom`, and `/car_state/odom`.
Override via env var:

```bash
ODOM_TOPIC=/my/odom python3 -m uvicorn main:app --host 0.0.0.0 --port 8082
```

---

## Ports

| Service | Port |
|---------|------|
| Backend API + WebSocket | `8082` |
| Frontend dev server | `5173` |
