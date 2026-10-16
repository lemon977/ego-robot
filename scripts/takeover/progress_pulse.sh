#!/usr/bin/env bash
# Retired: v13 is sealed. Refuse before touching any project file.
echo "LEGACY_TAKEOVER_HELPER_RETIRED: pulse; register a new finite Task Packet" >&2
exit 64
# Historical implementation below is evidence, not a launch instruction.
set -euo pipefail
cd /path/to/chaoyang
W=_run/caches/takeover_autopilot
D=docs/plans/EXACT78_TAKEOVER_20261004
PY=/usr/local/bin/python
mkdir -p "$W"
if [ -f "$W/pulse.pid" ] && grep -q progress_pulse.sh "/proc/$(cat "$W/pulse.pid")/cmdline" 2>/dev/null; then
  if [ "$(cat "$W/pulse.pid")" != "$$" ]; then
    echo "another pulse holds pid $(cat "$W/pulse.pid")"
    exit 0
  fi
fi
echo $$ > "$W/pulse.pid"
last=""
while true; do
  date -Iseconds > "$W/PULSE_HEARTBEAT"
  $PY - <<'PY' || true
import json
from datetime import datetime
from pathlib import Path
ROOT = Path("/path/to/chaoyang")
D = ROOT / "docs/plans/EXACT78_TAKEOVER_20261004"
W = ROOT / "_run/caches/takeover_autopilot"
task = "exact78_takeover_b2_full_v13"
prog_p = ROOT / f"_run/current/{task}/attempts/attempt_0001/candidate/PROGRESS.json"
if not prog_p.is_file():
    raise SystemExit(0)
prog = json.loads(prog_p.read_text(encoding="utf-8"))
result_p = ROOT / f"_run/current/{task}/attempts/attempt_0001/RESULT.json"
if result_p.is_file():
    status = str(json.loads(result_p.read_text(encoding="utf-8")).get("status") or "")
    if status == "PASSED" or status.startswith("FAILED"):
        raise SystemExit(0)
counts = prog.get("counts") or {}
ok = int(counts.get("OK") or 0)
stamp = f"{ok}|{counts}|{prog.get('updated_at')}"
marker = W / "pulse_last_v13.txt"
if marker.is_file() and marker.read_text() == stamp:
    raise SystemExit(0)
now = datetime.now().astimezone()
hm = now.strftime("%H:%M")
iso = now.isoformat(timespec="seconds")
elapsed = float(prog.get("elapsed_s") or 0)
eta = ""
if ok >= 2:
    remain = 234 - ok
    finish = datetime.fromtimestamp(now.timestamp() + elapsed / ok * remain).astimezone().strftime("%m-%d %H:%M")
    eta = f"按当前速度约 {finish} 跑完。"
line = (f"- {hm} [pulse] `{task}` 进度 {counts}，已运行 {elapsed/60:.0f} min，"
        f"PROGRESS 更新于 {prog.get('updated_at')}。{eta}")
prog_doc = D / "PROGRESS_ZH.md"
text = prog_doc.read_text(encoding="utf-8")
if line.split("进度", 1)[-1] not in text[-2000:]:
    prog_doc.write_text(text + line + "\n", encoding="utf-8")
state_p = D / "STATE.json"
state = json.loads(state_p.read_text(encoding="utf-8"))
state["updated_at"] = iso
state["steps"]["m9-full"] = (
    f"RUNNING {task} (pulse {iso}): counts {counts}, elapsed {elapsed/60:.0f} min; "
    f"v12 remains the sealed baseline (E0020) until v13 seals. {eta}"
)
tmp = state_p.with_name(state_p.name + ".pulse.tmp")
tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
tmp.replace(state_p)
marker.write_text(stamp)
print(line)
PY
  sleep 900
done
