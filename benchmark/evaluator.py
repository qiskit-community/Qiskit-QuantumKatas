"""Code evaluation for quantum katas benchmark."""

import ast
import io
import multiprocessing
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    """Result of evaluating a generated solution."""

    passed: bool
    syntax_valid: bool
    test_passed: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stdout: Optional[str] = None


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response.

    Handles markdown code blocks, triple-quoted strings, and raw code.
    Also handles malformed responses with unclosed code blocks.
    """
    response = response.strip()

    # Try to extract from markdown code block (properly closed)
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Handle unclosed code blocks (model started code block but didn't close it)
    # This catches cases like: ```python\ncode... (no closing ```)
    unclosed_patterns = [
        r"```python\n(.*)$",
        r"```py\n(.*)$",
        r"```\n(.*)$",
    ]

    for pattern in unclosed_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Handle response wrapped in triple quotes (common with completion models)
    if response.startswith('"""') and response.endswith('"""'):
        return response[3:-3].strip()
    if response.startswith("'''") and response.endswith("'''"):
        return response[3:-3].strip()

    # If no code block, return as-is (might be raw code)
    return response


def check_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _grade_worker(code: str, test_code: str, entry_point: str, timeout: float, result_queue):
    """Run the isolated grader in its own process, so a hung or crashing test can't
    take the runner down. The grader runs the candidate in a further process that never
    sees the test source; see benchmark/isolation.py."""
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    try:
        from benchmark.isolation import grade

        result = grade(code, test_code, entry_point, timeout)
    except BaseException as e:  # noqa: BLE001
        result = {"passed": False, "error_type": type(e).__name__, "error": str(e)}
    finally:
        sys.stdout = old_stdout
    result["stdout"] = stdout_capture.getvalue()
    result_queue.put(result)


def run_test(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 30.0,
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Run test code against solution with timeout.

    The candidate never runs in the process that executes the test or reports the
    verdict, so it can't patch the checker or report its own pass.

    Returns: (passed, error_message, error_type, stdout)
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    process = ctx.Process(
        target=_grade_worker,
        args=(code, test_code, entry_point, timeout, result_queue),
    )
    process.start()
    # Each candidate call has its own `timeout`; the whole test gets a wider bound.
    process.join(timeout=timeout * 2)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"Timeout after {timeout * 2}s", "Timeout", None

    if result_queue.empty():
        return False, "Process ended without result", "ProcessError", None

    result = result_queue.get()
    error_type = result.get("error_type")
    error = result.get("error")
    error_message = None
    if not result["passed"]:
        error_message = f"{error_type}: {error}" if error_type == "AssertionError" else error
    return result["passed"], error_message, error_type, result.get("stdout")


def evaluate_solution(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 30.0,
) -> EvaluationResult:
    """Evaluate a generated solution.

    Args:
        code: The generated code from the LLM
        test_code: The test code to run
        entry_point: The expected function name
        timeout: Maximum execution time in seconds

    Returns:
        EvaluationResult with pass/fail status and details
    """
    # Extract code from response
    extracted_code = extract_code_from_response(code)

    # Check syntax
    syntax_valid, syntax_error = check_syntax(extracted_code)
    if not syntax_valid:
        return EvaluationResult(
            passed=False,
            syntax_valid=False,
            test_passed=False,
            error_message=syntax_error,
            error_type="SyntaxError",
        )

    # Run tests
    test_passed, error_message, error_type, stdout = run_test(
        extracted_code,
        test_code,
        entry_point,
        timeout,
    )

    return EvaluationResult(
        passed=test_passed,
        syntax_valid=True,
        test_passed=test_passed,
        error_message=error_message,
        error_type=error_type,
        stdout=stdout,
    )


def validate_dataset_solution(
    solution: str,
    test_code: str,
    entry_point: str,
) -> bool:
    """Quick validation of a dataset solution (for testing dataset quality)."""
    result = evaluate_solution(solution, test_code, entry_point)
    return result.passed
