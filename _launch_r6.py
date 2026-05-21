"""Launcher that fully detaches the r6 training subprocess.

`subprocess.Popen(..., start_new_session=True)` is Python's portable
equivalent of `setsid` — creates a new process group so the parent's
process-group SIGTERM/SIGHUP at agent-call teardown doesn't reach the child.
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path("/Users/kaede/tts/_sovits_mlx_train/r6")
OUT.mkdir(parents=True, exist_ok=True)

log_path = OUT / "launch.log"
pid_path = OUT / "pid"

cmd = [
    "/Users/kaede/tts/GPT-SoVITS/.venv/bin/python",
    "-u",
    "/Users/kaede/tts/_sovits_mlx/train_r6.py",
    "--batch-size", "8",
    "--output-dir", str(OUT),
]

log_fh = open(log_path, "ab")
proc = subprocess.Popen(
    cmd,
    cwd="/Users/kaede/tts/_sovits_mlx",
    stdout=log_fh,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,        # NEW process group, immune to SIGHUP
    close_fds=True,
)
pid_path.write_text(str(proc.pid) + "\n")
print(f"launched PID: {proc.pid}")
print(f"log:  {log_path}")
print(f"pid:  {pid_path}")
