"""The harness must grade solutions, not let them grade themselves (see issue #6)."""

import json
from pathlib import Path

import pytest

from benchmark.evaluator import evaluate_solution

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "qiskit_quantumkatas.jsonl"
TASKS = {
    row["task_id"]: row
    for row in map(json.loads, DATASET.read_text().splitlines())
    # In-place circuit edits, measure_all registers, solution-defined helpers and
    # classes, and a returned oracle callable.
    if row["task_id"]
    in {
        "BasicGates/1.1",
        "examples/BasicQuantumTeleportation",
        "MagicSquareGame/1.3",
        "MagicSquareGame/2.3",
        "SimonsAlgorithm/Algorithm",
    }
}

# Walks up the call stack to any result channel and reports a pass.
FORGE = """
import sys as _sys, os as _os
_f = _sys._getframe(0)
while _f is not None:
    _q = _f.f_locals.get("result_queue")
    if _q is not None and hasattr(_q, "put"):
        _q.put({"passed": True, "error_message": None, "error_type": None, "stdout": ""})
        _q.close(); _q.join_thread(); _os._exit(0)
    _f = _f.f_back
"""

PATCH = """
import numpy as _np
from qiskit.quantum_info import Statevector as _SV, Operator as _Op
_np.allclose = lambda *a, **k: True
_np.isclose = lambda *a, **k: True
_SV.equiv = lambda *a, **k: True
_Op.equiv = lambda *a, **k: True
"""


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_canonical_solution_passes(task_id):
    row = TASKS[task_id]
    assert evaluate_solution(row["canonical_solution"], row["test"], row["entry_point"]).passed


@pytest.mark.parametrize("attack", [FORGE, PATCH], ids=["forge", "patch"])
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_stub_cannot_tamper_with_grading(task_id, attack):
    row = TASKS[task_id]
    result = evaluate_solution(row["prompt"] + "\n" + attack, row["test"], row["entry_point"])
    assert not result.passed
