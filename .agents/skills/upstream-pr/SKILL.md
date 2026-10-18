---
name: upstream-pr
description: Prepare a Buddy-MLIR commit and pull request - scope, tests, docs, formatting, commit message, PR description with measurements - and handle review feedback. Use when committing, opening or updating a PR, or answering a reviewer.
---

# Upstream PR

## Scope

- One mechanism per PR. A PR that needs another unmerged PR waits for it, or
  says so at the top of its description.
- Base the branch on the latest `main`; rebase when `main` moves and drop
  commits that were merged (squash merges rewrite them).
- No unrelated refactoring, renames or reformatting.

## Before committing

1. Build and run the tests: `ninja -C build check-buddy`, plus the tests of
   the touched area (e.g. `tests/Python/test_k3_w4_*.py`,
   `tests/Runtime/...`). Add or extend a test for new behavior; a test of a
   contract must fail without the change (check it).
2. Formatting on the changed files only:
   `pre-commit run --files <changed files>`.
3. Update the docs the change touches (`docs/*.md`), including numbers that
   the change makes stale.
4. Do not commit the `llvm` submodule pointer, build directories, local
   scripts, secrets or machine-specific paths. Stage files explicitly.

## Commit message

```
[area] Imperative summary of the change

What was wrong or slow and why (the mechanism, with the evidence).
What the change does. Whether results are bit-identical or change, and
how that was checked. Measured effect.
```

Areas used in the history: `[tools]`, `[runtime]`, `[frontend]`,
`[midend]`, `[models]`, `[docs]`, combinations like `[frontend][runtime]`.

## PR description

Follow `.github/PULL_REQUEST_TEMPLATE.md`:

- **Summary**: the problem, the cause, the change (a table of files /
  changes for multi-part PRs), alternatives considered and why not.
- **Results**: baseline vs PR (and the reference framework if relevant) per
  input size, how measured (board, command, runs), whether the output is
  identical.
- **Validation**: tests run and their counts, accuracy checks for numerics
  changes.
- **Checklist** from the template, ticked honestly.

## Review feedback

- Verify each point against the code; fix what is right, explain with
  evidence what is not.
- Push fixes as new commits on the PR branch (do not force-push away the
  history the reviewer saw unless asked), re-run the tests, and summarize
  what changed in reply.
- When a fix could change performance (e.g. alignment), measure it again.
