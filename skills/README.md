# skills

Repository-local agent skills, in the same `SKILL.md` format as the IX83 bundle: YAML
frontmatter (`name`, `description`) plus scope, preconditions, procedure, verification
and what the skill must not be used to claim.

A skill is guidance for a class of task. It is not a recipe, a run record or an
authorization: a recipe is a versioned executable workflow, a run record is resolved
evidence, and hardware work still needs the authorization its own record names. See [the
handbook](../docs/agent-handbook.md).

| Skill | Use when |
|---|---|
| [offline-scientific-processing](offline-scientific-processing/SKILL.md) | Running a hardware-free reconstruction or replay from recorded data and reporting what it establishes |
| [windows-sandbox-validation](windows-sandbox-validation/SKILL.md) | Composing, running and interpreting a Windows test or validation command in a DSH session |
| [work-package-delegation](work-package-delegation/SKILL.md) | Delegating a bounded work package with exclusive write ownership, frozen contracts and explicit evidence classes |

## Adding a skill

Name the directory for the task class in kebab case and match the frontmatter `name`;
keep `description` to one sentence stating trigger and outcome; keep run-specific
numbers in the record that measured them; link relatively; and never assert an
acceptance state a record has not established.
