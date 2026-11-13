---
name: work-package-delegation
description: Delegate or accept a bounded AHT work package with exclusive write ownership, frozen producer/consumer contracts and explicit evidence classes.
---

# Work-package delegation

## Scope

Assigning a bounded unit of work to another agent, and accepting its result. Adapted
from section 8 of the AHT/IX83 convergence plan
(`02_docs_offline/aht_ix83_convergence_plan.md` in the ARH instrument record); it
restates that process, invents no second one and grants no authority the plan withholds.
Read [AGENTS.md](../../AGENTS.md) and [the handbook](../../docs/agent-handbook.md)
first.

## Preconditions

- The work maps to one work package, or one reviewable slice, with acceptance
  criteria checkable from artifacts.
- Dependencies are named: accepted ones with evidence, pending ones with what
  they block. A dependent change cannot be accepted while one is pending.
- The coordinator owns shared files — `pyproject.toml`, `uv.lock`, plan maps,
  schemas, shared fixtures. Two agents never edit them in the same round.

## Procedure

1. **Assign exclusive write ownership** as explicit files and directories;
   everything else is read-only for that delegate. "The processing area" is not an
   ownership grant.
2. **Freeze the contract before the work**: the versioned interface a consumer
   reads — types, units, coordinate names, partial/error semantics — and its
   compatibility surface. Changing it later is a new decision.
3. **Fill the delegation prompt completely:**

   ```text
   Implement WP<id> from 02_docs_offline/aht_ix83_convergence_plan.md.
   Objective: <bounded outcome>.
   Owning repository/worktree: <path, branch, starting revision>.
   Exclusive writable paths: <explicit files/directories>.
   Read-only references: <paths and exact revisions>.
   Dependencies accepted: <WP ids, contract versions, evidence links>.
   Inputs still pending: <items; dependent changes must wait>.
   Preserve: <public APIs, metadata, numerical behavior, cleanup behavior>.
   Deliver: <patch, documentation, tests/evidence, compatibility notes>.
   Acceptance: <specific checks for this work package>.
   Read AGENTS.md and the owning test instructions first.
   Inspect current status; preserve user work. Do not touch other agents' paths.
   Do not publish, push, merge, change pins or operate hardware unless this task
   explicitly authorizes that specific action.
   Report interface conflicts before changing a shared schema or dependency.
   Finish with changed paths, checks/results, provenance, open issues, next owner.
   ```

4. **State the evidence class up front** — `documented`, `source-present`,
   `freshly-verified`, `hardware-qualified`
   ([handbook](../../docs/agent-handbook.md#3-status-vocabulary)) — so the result
   cannot be over-read.
5. **Require the completion report:** changed paths; commands with counts; source
   and dependency revisions; artifacts and digests; what was not attempted and
   why; interface changes or conflicts; remaining gates; next owner.
6. **Verify before accepting.** Re-open the changed files and artifacts; a
   delegate's summary is a claim until the diff and the file confirm it, and a
   numerical claim is re-read from the report, not the message.
7. **Integrate deliberately**: dependency and lock changes route through the
   coordinator, and the instrument record's submodule pointer is updated
   separately, after the owning repository is tested.

## Hardware authorization boundary

Documentation, refactoring, simulation and recorded-data work never authorize
connecting, moving, illuminating, flashing firmware or changing physical safety state. A
package needing a device is refused or deferred until the instrument record names the
authorization, the envelope and the operator. One process holds a device handle;
delegates receive exported files, never a handle, and `hardware-qualified` evidence can
only come from that authorized run.

## Verification

Every acceptance criterion maps to a command or artifact produced in the reporting
round; the changed-path list matches the ownership grant, and anything outside it is
reported as a violation rather than merged quietly; the implemented contract is the
frozen one, or the deviation is written down with its consumer impact.

## What this skill must not be used to claim

- That a work package is authorized for hardware, publication, dependency changes
  or Git integration; only the task's own authorization does that.
- That a delegated result is accepted because it was reported — acceptance needs
  the artifact.
- That parallel work on a shared file was safe because no conflict was noticed.
- That a plan's `planned` state became `accepted` without evidence.
