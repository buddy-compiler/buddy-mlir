# Buddy-MLIR Governance

## 1. Project Mission

Buddy-MLIR develops an open compiler framework connecting PyTorch, MLIR and diverse hardware for compiler research, co-design and reproducible model execution. This charter governs buddy-compiler/buddy-mlir.

## 2. Roles

1. **Project leader:** makes final technical decisions after consultation, delegates duties and arranges succession.

2. **Maintainer:** maintains assigned code, reviews, tests, documentation, compatibility and releases.

3. **Contributor:** contributes code, documentation, tests, reviews or reports; anyone may propose and discuss changes.

### Current assignments

This draft designates **@zhanghb97 as project leader**. Maintenance responsibilities follow the paths listed in repository CODEOWNERS:

| Account | Responsibility |
| --- | --- |
| @zhanghb97 | Project leader, project coordination and default ownership; frontend, midend and backend maintenance |
| @R-Tars | frontend and midend maintenance/review |
| @WuXintong123 | midend and backend maintenance/review |

## 3. Decision-Making

1. Routine public PRs require one qualified human approval independent of the author and passing mandatory CI.

2. Major architecture, membership and support-policy changes receive at least seven days of public Issue/RFC discussion, followed by an explicit project leader decision with reasons. Silence is not approval.

3. Stable releases require two distinct maintainers' approvals, clean installation of actual artifacts, import/CLI/minimal-model verification, and published dependencies, support matrix and results.

## 4. Becoming a Maintainer

Maintainers may nominate sustained contributors with technical judgment and collaborative conduct; non-code work counts equally. After candidate acceptance and seven days of public consultation, the project leader confirms appointment. Update the roster, CODEOWNERS and permissions together.

## 5. Stepping Down

1. Maintainers may resign at any time and become emeritus.

2. After six months without meaningful maintenance, contact the person, allow at least 30 days to respond and remind them seven days before the deadline. Reviews, documentation and mentoring count; approved leave pauses inactivity.

3. The person may continue with a review within 90 days, take time-limited leave with coverage, or retire voluntarily.

4. Hand over work, update the roster/CODEOWNERS and revoke role-derived access within seven days; rotate credentials where needed. Preserve attribution.

## 6. Conflict Resolution

1. Discuss technical differences publicly; the project leader resolves unresolved issues. Anyone may request reconsideration with new evidence. Technical disagreement is not misconduct.

2. Matters involving the project leader, specific personal financial interests or conduct complaints require two uninvolved maintainers, or an agreed mediator if unavailable.

3. Exceptional removal for serious abuse or persistent incapacity requires independent review, at least 14 days to respond.

## 7. Organizational Context

Technical responsibility rests with individuals; employment, sponsorship and donations grant no privilege. This charter transfers no assets and establishes no foundation hosting.

## 8. Code of Conduct

1. The project follows **Contributor Covenant 2.0** in the repository's [CODE_OF_CONDUCT.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/CODE_OF_CONDUCT.md), consistent with CONTRIBUTING.md. It applies in community spaces and when officially representing the community. Respectful, inclusive and constructive participation is expected; harassment, discrimination, insults and unauthorized disclosure of private information are prohibited.

2. Report conduct incidents privately to **hongbin2019@iscas.ac.cn** under the current policy. Community leaders investigate promptly and fairly and protect reporter privacy.

3. Depending on community impact, measures include correction, warning, temporary ban or permanent ban. CODE_OF_CONDUCT.md governs the procedure and severity; the project leader is also bound by it.

4. Report vulnerabilities separately under [SECURITY.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/SECURITY.md) to the same email. Keep vulnerability disclosure, conduct reports and technical disagreements distinct.

5. If application requirements call for changes, update the code of conduct and contribution guide together through a separate PR.

## 9. Changes to Governance

1. Initial adoption requires seven days of public review, approval by existing authorized repository owners, and acceptance by the project leader and at least two maintainers.

2. Later amendments require a PR, 14 days of discussion and explicit approval by an unconflicted project leader.

3. Publish the effective date and synchronize the roster, related documents and permissions.

## 10. AI-Assisted Governance

1. AI may assist code, tests, review, triage, discussion summaries and RFCs. A named human must understand, verify and take responsibility.

2. Disclose material assistance, tool, scope and actual checks in the PR/report; trivial spelling corrections need no individual disclosure.

3. AI holds no maintainer status and cannot replace independent human approval, community consensus or project leader decisions. It cannot autonomously appoint/retire people, adjudicate appeals, change permissions or approve releases.

4. Compiler transformations need regression tests and trusted-reference comparisons. Never fabricate tests, performance or citations; identify unexecuted checks.

5. Protect credentials, private code, vulnerabilities, models and personal data; do not send them to external AI services without authorization.

6. Check provenance, licenses and attribution.

7. Project agents need a human owner, least privilege, data/cost limits, audit and disable controls. Privileged or non-public-data access requires RFC review, keeping sensitive details private.

8. Review quarterly, allow human correction of automated recommendations and never sanction solely on an AI-detector score.
