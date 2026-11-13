---
name: windows-sandbox-validation
description: Compose, run and interpret a Windows test or validation command in a DSH session, keeping environment faults separate from test results.
---

# Windows sandbox validation

## Scope

Running tests, linters and bounded validation commands on this Windows host and
reporting what they establish. Covers DSH agent sessions and a normal host PowerShell;
not instrument rehearsal or hardware qualification. Read [AGENTS.md](../../AGENTS.md),
[Windows test execution](../../docs/windows_test_execution.md) and
[testing.md](../../docs/testing.md) before composing a command.

## Preconditions

- A non-elevated PowerShell session; never run pytest elevated.
- A `.venv` that imports `redsun_aht`.
- A stated purpose — gate, regression check or spot check — which decides the
  selection and what the result may be called.
- No other process writing the same `tmp/` run root or holding a device the
  selection needs.

## Procedure

1. **Use the runner, not an ad-hoc pytest line.** It supplies the fresh basetemp,
   `mkdir` shim, napari cache override, `pymmcore_plus` log path and CuPy caches.

   ```powershell
   pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests/test_multilayer.py -q
   ```

2. **Pick the narrowest selection that answers the question** and name it in the
   report; a whole-suite pass is not needed for a documentation change, and one
   file is not evidence for the suite.
3. **Read the failure before calling it a failure.** Per the classification table
   in [Windows test execution](../../docs/windows_test_execution.md#environment-fault-or-test-failure),
   fixture-setup `PermissionError`, cache denials and CuPy stalls are
   environmental; an executed assertion or numerical comparison is a test
   failure.
4. **Re-run an environmental failure once with the recipe applied.** If it
   persists, report the command, error and path. Do not rebuild the environment,
   patch `site-packages` or elevate.
5. **Record the result with its exclusions.** `research`, `integration`,
   `hardware` and `catalog` groups are excluded by default; a run that excluded
   them is not a full-suite pass.

## Verification

The run prints its run root and exit code; keep the log path as the artifact. Reported
counts match the summary exactly (`N passed`, `M skipped`, `K warnings`) for the stated
selection. For a changed code path the guarding test is demonstrably sensitive to the
change, and the report says which one. Every claim about the run points at a command
that can be re-executed.

## When a failure is environmental

Report the command, run root, exception type and message, the path it named, and the
recipe that addresses it. Do not promote it to a code defect, mark the work blocked
without a second attempt, delete a protected directory as an agent, or describe a
skipped test as passed.

## What this skill must not be used to claim

- Hardware qualification, physical acceptance or instrument readiness.
- That a green selection covers behavior it never executed, or that a passing
  simulation proves a device path.
- That an environment limitation is a property of the repository rather than of
  this host, account or session token.
- That a test's summary replaces reading the assertion that failed.
