"""Grade one Qiskit QuantumKatas solution with the candidate isolated from the test.

The original benchmark harness execs the candidate and the test in one process, so a
solution can tamper with grading: it can monkeypatch the checker (e.g. `numpy.allclose`),
or walk up the call stack to the harness's result queue, report a pass and exit (which
passes all 350 tasks). Here the candidate runs in a spawned worker that never sees the
test source and never decides the verdict. The test runs in this (trusted) process
against a proxy of the entry point:

- arguments are sent to the worker with cloudpickle (test-defined oracles included);
- in-place changes the candidate makes to its arguments (most tasks mutate the
  QuantumCircuit they receive) are copied back onto the caller's objects;
- returned callables (oracle-building tasks) stay in the worker and are called remotely,
  so candidate code never executes in the trusted process.

Used by `benchmark.evaluator.run_test`; `grade()` returns {"passed": bool, ...}.
"""

import multiprocessing
import traceback

import cloudpickle

OUTER_GRACE = 5.0


# Top-level packages the trusted side may import by name when a test uses a symbol
# that only the solution imported (e.g. `Pauli`). Anything else stays remote.
IMPORTABLE = {
    "qiskit", "qiskit_aer", "numpy", "scipy", "math", "cmath", "random", "itertools",
    "functools", "collections", "typing", "fractions", "operator",
}


# ---------------------------------------------------------------- worker (untrusted)
def manifest(namespace: dict, entry_point: str, handles: dict) -> dict:
    """Describe the solution's public names so tests that call its helpers still work.

    Some tests call helpers the solution defines (e.g. `bob_classical`) or use names
    only the solution imported (e.g. `Pauli`). Library symbols are re-imported on the
    trusted side; anything the solution defined itself stays in the worker.
    """
    import types

    out: dict = {}
    for name, value in namespace.items():
        if name.startswith("_") or name == entry_point:
            continue
        if isinstance(value, types.ModuleType):
            out[name] = ("module", value.__name__)
            continue
        module = getattr(value, "__module__", None) or ""
        qualname = getattr(value, "__qualname__", None)
        if module != "candidate" and qualname and "<" not in qualname:
            out[name] = ("import", module, qualname)
        elif isinstance(value, type):
            # Solution-defined data types (e.g. a `Pauli` enum the test indexes) travel by
            # value. This is the one path where solution code can run on the trusted side.
            out[name] = ("value", value)
        elif callable(value):
            handles[id(value)] = value
            out[name] = ("remote", id(value))
    return out


def serve_candidate(conn, solution: str, entry_point: str) -> None:
    import contextlib
    import io

    handles: dict[int, object] = {}

    def export(value):
        """Keep callables in the worker; send a handle instead."""
        if callable(value) and not isinstance(value, type):
            handles[id(value)] = value
            return ("__remote_callable__", id(value))
        if isinstance(value, tuple):
            return tuple(export(v) for v in value)
        if isinstance(value, list):
            return [export(v) for v in value]
        return value

    try:
        namespace: dict = {"__name__": "candidate"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(solution, namespace)
        if entry_point not in namespace:
            raise NameError(f"Entry point '{entry_point}' not found in solution")
        handles[0] = namespace[entry_point]
        conn.send_bytes(cloudpickle.dumps(("ready", manifest(namespace, entry_point, handles))))
    except BaseException as error:  # noqa: BLE001
        conn.send_bytes(cloudpickle.dumps(("error", f"{type(error).__name__}: {error}")))
        return

    while True:
        try:
            handle, args, kwargs = cloudpickle.loads(conn.recv_bytes())
        except EOFError:
            return
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = handles[handle](*args, **kwargs)
            # Pickle result and args together so identity (`return qc`) survives.
            out = ("ok", (export(result), args, kwargs))
            payload = cloudpickle.dumps(out)
        except BaseException as error:  # noqa: BLE001
            try:
                payload = cloudpickle.dumps(("raise", error))
            except Exception:  # noqa: BLE001 - unpicklable exception
                payload = cloudpickle.dumps(("raise", RuntimeError(repr(error))))
        conn.send_bytes(payload)


# ---------------------------------------------------------------- trusted side
def sync_circuit(original, updated) -> None:
    """Rebuild `original` from `updated` through Qiskit's public API.

    Copying `__dict__` is not enough: the Rust-backed circuit data then breaks methods
    like `remove_final_measurements(inplace=True)`. Registers the candidate added (e.g.
    the `meas` register from `measure_all`) are recreated, then the instructions are
    re-appended with bits mapped by index.
    """
    have = {r.name for r in original.qregs} | {r.name for r in original.cregs}
    for reg in [*updated.qregs, *updated.cregs]:
        if reg.name not in have:
            original.add_register(type(reg)(reg.size, reg.name))
    if (original.num_qubits, original.num_clbits) != (updated.num_qubits, updated.num_clbits):
        raise RuntimeError("cannot mirror the candidate's register changes onto the circuit")
    original.clear()
    for inst in updated.data:
        original.append(
            inst.operation,
            [original.qubits[updated.find_bit(q).index] for q in inst.qubits],
            [original.clbits[updated.find_bit(c).index] for c in inst.clbits],
            copy=False,
        )
    original.global_phase = updated.global_phase
    original.metadata = updated.metadata


def sync_in_place(original, updated) -> None:
    """Copy the worker's mutations of an argument back onto the caller's object."""
    from qiskit import QuantumCircuit

    if original is updated:
        return
    if isinstance(original, QuantumCircuit) and isinstance(updated, QuantumCircuit):
        sync_circuit(original, updated)
    elif isinstance(original, list) and isinstance(updated, list):
        original[:] = updated
    elif isinstance(original, dict) and isinstance(updated, dict):
        original.clear()
        original.update(updated)
    elif type(original).__name__ == "ndarray" and getattr(original, "flags", None) is not None:
        if original.flags.writeable:
            original[...] = updated


class Remote:
    def __init__(self, conn, timeout: float):
        self.conn = conn
        self.timeout = timeout

    def call(self, handle: int, args, kwargs):
        self.conn.send_bytes(cloudpickle.dumps((handle, args, kwargs)))
        if not self.conn.poll(self.timeout):
            raise TimeoutError("candidate call timed out")
        status, value = cloudpickle.loads(self.conn.recv_bytes())
        if status == "raise":
            raise value
        result, new_args, new_kwargs = value
        mapping = {}
        for old, new in zip(args, new_args):
            sync_in_place(old, new)
            mapping[id(new)] = old
        for key, old in kwargs.items():
            sync_in_place(old, new_kwargs[key])
            mapping[id(new_kwargs[key])] = old
        return self.wrap(result, mapping)

    def wrap(self, value, mapping):
        if id(value) in mapping:
            return mapping[id(value)]
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "__remote_callable__":
            handle = value[1]
            return lambda *a, **k: self.call(handle, a, k)
        if isinstance(value, tuple):
            return tuple(self.wrap(v, mapping) for v in value)
        if isinstance(value, list):
            return [self.wrap(v, mapping) for v in value]
        return value


def resolve_manifest(entries: dict, remote: Remote) -> dict:
    import importlib

    resolved: dict = {}
    for name, entry in entries.items():
        kind = entry[0]
        if kind in ("module", "import") and entry[1].split(".")[0] not in IMPORTABLE:
            continue  # never import a module the candidate chose from outside the allowlist
        try:
            if kind == "module":
                resolved[name] = importlib.import_module(entry[1])
            elif kind == "import":
                value = importlib.import_module(entry[1])
                for part in entry[2].split("."):
                    value = getattr(value, part)
                resolved[name] = value
            elif kind == "value":
                resolved[name] = entry[1]
            elif kind == "remote":
                handle = entry[1]
                resolved[name] = lambda *a, _h=handle, **k: remote.call(_h, a, k)
        except (ImportError, AttributeError):
            continue
    return resolved


def grade(solution: str, test: str, entry_point: str, timeout: float) -> dict:
    ctx = multiprocessing.get_context("spawn")
    conn, worker_conn = ctx.Pipe()
    worker = ctx.Process(target=serve_candidate, args=(worker_conn, solution, entry_point))
    worker.start()
    worker_conn.close()
    try:
        if not conn.poll(timeout + OUTER_GRACE):
            return {"passed": False, "error_type": "Timeout", "error": "worker startup"}
        status, detail = cloudpickle.loads(conn.recv_bytes())
        if status == "error":
            return {"passed": False, "error_type": "SolutionError", "error": detail}

        remote = Remote(conn, timeout)
        namespace: dict = {"__name__": "katas_test"}
        namespace.update(resolve_manifest(detail, remote))
        namespace[entry_point] = lambda *a, **k: remote.call(0, a, k)
        exec(test, namespace)
        # Re-bind after exec in case the test imported a same-named symbol.
        namespace[entry_point] = lambda *a, **k: remote.call(0, a, k)
        name = f"test_{entry_point}"
        if name not in namespace:
            candidates = [k for k in namespace if k.startswith("test_")]
            if not candidates:
                return {"passed": False, "error_type": "MissingTest", "error": ""}
            name = candidates[0]
        namespace[name]()
        return {"passed": True}
    except AssertionError as error:
        return {"passed": False, "error_type": "AssertionError", "error": str(error)[:500]}
    except BaseException as error:  # noqa: BLE001
        return {
            "passed": False,
            "error_type": type(error).__name__,
            "error": traceback.format_exc(limit=3)[-800:],
        }
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join()
        conn.close()
