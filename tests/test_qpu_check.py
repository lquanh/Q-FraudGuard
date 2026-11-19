import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize(("solver", "count", "exit_code", "primary"), [
    ("exact", 1, 0, 1),
    ("exact", 2, 1, 1),        # one fewer window than requested
    ("quafu-mock", 1, 1, 0),   # primary rejects the window, CPU fallback succeeds
])
def test_deployment_check_requires_requested_backend_and_full_sample(tmp_path, solver, count, exit_code, primary):
    scores = tmp_path / "scores.pkl"
    data = [dict(transaction_id=str(i), timestamp=i, risk_score=1 / (i + 1),
                 sender_id="shared", receiver_id=str(i)) for i in range(8)]
    pd.DataFrame(data).to_pickle(scores)
    script = Path(__file__).resolve().parents[1] / "scripts/qpu_check.py"
    result = subprocess.run(
        [sys.executable, str(script), "--solver", solver, "--windows", str(count),
         "--window-size", "8", "--fallback", "sqa", "--scores", str(scores), "--out", str(tmp_path)],
        env={**os.environ, "QFG_QUAFU_MAX_QUBITS": "4"}, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == exit_code, result.stderr
    report = json.loads(next(tmp_path.glob("qpu_check_*.json")).read_text(encoding="utf-8"))
    assert report["passed"] is (exit_code == 0)
    assert report["primary_windows"] == primary
