# KonveyN2AI BigQuery Backend Constitution

## Core Principles

### I. Library-First Delivery
- Every feature begins as a standalone library under `src/` with a single, opinionated purpose.
- Libraries must ship with `llms.txt` usage notes, documented public API, and dependency diagrams before implementation work starts.
- Application entrypoints (CLI, services, notebooks) may only consume published library contracts—no direct feature code outside libraries.
- When a capability spans domains, split it into cooperating libraries instead of widening scope; justify exceptions in `plan.md` Complexity Tracking.

### II. CLI Interface & Text Protocol
- Each library exposes a CLI surface in `src/cli/` that mirrors library capabilities end-to-end.
- Commands must support `--help`, `--version`, and `--format {human,json}` switches and honour stdout for success, stderr for diagnostics, exit codes for state.
- Prefer piping and file descriptors over interactive prompts; provide sample sessions in `quickstart.md`.
- CLI docs must cross-reference the Specify CLI (`/specify`, `/plan`, `/tasks`) touchpoints so agents know when to call which command.

### III. Test-First (NON-NEGOTIABLE)
- Red → Green → Refactor is enforced: write or update the failing test before touching implementation files.
- Test order: contract tests, then integration tests, then end-to-end validations, then unit tests for edge cases.
- Real dependencies must be used in tests (BigQuery datasets, Google auth flow, filesystem). Mocks or fakes are allowed only when exercising the real dependency would cause irreversible side effects or violate compliance; each exception must include a justification comment and a link to the mitigation plan in the test file.
- Commits must show failing test introduction prior to fix; CI blocks merges that skip the red phase.

### IV. Integration Testing & Real Environments
- Any new library, contract change, or shared schema update triggers integration tests against the BigQuery staging dataset and real service accounts.
- Maintain reusable fixtures for dataset provisioning via `make setup`; tests clean up after themselves.
- Integration suites run on every merge request; failures halt deployments until resolved.
- Document integration triggers and coverage in `tests/README.md` for traceability.

### V. Observability & Telemetry
- All libraries emit structured logs (JSON) with correlation IDs, request context, and BigQuery job reference IDs.
- CLI commands mirror log output with `--format json` for agents; human-readable output must still include trace IDs.
- Capture metrics for query latency, dataset mutations, and CLI execution time; store dashboards or queries in `docs/observability/`.
- Error handlers must attach remediation links or runbook references.

### VI. Versioning & Change Management
- Adopt MAJOR.MINOR.BUILD (build increments per merge). Declare current version in this constitution and mirror it in template footers.
- Breaking changes require dual-run strategy: maintain prior contract behind feature flag until migration tasks are complete.
- Update migration notes (`docs/migrations/`) whenever schemas, CLI flags, or library interfaces change.
- The `plan-template.md` Constitution Check must verify version increments and migration planning before work starts.

### VII. Simplicity & Focus
- Default to a single project layout (`src/`, `tests/`) unless Technical Context mandates web or mobile split; never exceed three top-level code projects without a documented exception.
- Ban heavyweight patterns (Repository, Unit of Work, CQRS) unless a production incident or compliance need demands them; capture justification in Complexity Tracking.
- Prefer direct use of frameworks and SDKs over wrappers; if a wrapper is unavoidable, document its lifecycle and teardown strategy.
- Always challenge scope creep—if a feature adds unrelated capabilities, spin a new specification instead of overloading the current one.

## Specify CLI Alignment

- `specify init <project>` bootstraps repositories; honour `--ai`, `--script`, `--ignore-agent-tools`, `--no-git`, `--here`, `--skip-tls`, and `--debug` options when scripting automation or documenting setup.
- `specify check` must pass before allowing `/plan` or `/tasks`; block work until required agent tooling (Claude Code, Gemini CLI, Cursor, Opencode, Copilot) is available.
- On initialization, ensure the generated branch, specification directory, and templates remain in sync with this constitution and update checklist.
- Record CLI invocation examples in onboarding docs so agents can reproduce bootstrap flows verbatim.

## Development Workflow

- **Backlog Alignment (Prerequisite)**: Before invoking `/specify`, `/plan`, or `/tasks`, audit the most recent spec, plan, tasks, CLAUDE guidance, and open pull requests. Carry forward incomplete items, TODOs, or checklist failures and document how they will be resolved before advancing phases.

1. **/specify – Functional Intent**: Capture business goals, user stories, and acceptance criteria without dictating technology choices. Use follow-up prompts to clarify requirements and validate checklists.
2. **/plan – Technical Blueprint**: Define stack decisions, research needs, data models, contracts, and quickstarts. Apply the Constitution Check before Phase 0 and after Phase 1; refuse to proceed if violations persist.
3. **/tasks – Execution Breakdown**: Generate ordered task lists (TDD-first) using the template rules; mark `[P]` only for file-isolated tasks.
4. **Implementation Phases**: Execute tasks sequentially, maintain red tests before green commits, and continuously update observability artefacts.
5. **Validation & Release**: Run contract, integration, and performance suites; document migrations; increment BUILD number; update dashboards.
6. **Review Loop**: On each amendment, run the Constitution Update Checklist and synchronize plan/task templates, `CLAUDE.md`, and related command docs.

## Governance

- This constitution supersedes all other process documentation. Any conflicting guideline must either be updated or explicitly superseded here.
- Amendments require: (a) running `/scripts/check-task-prerequisites.sh` or equivalent to ensure dependencies are current, (b) updating all checklist-linked templates, and (c) recording changes in `docs/migrations/CHANGELOG.md`.
- Pull requests must cite constitutional compliance in their description and reference affected principles.
- Runtime guidance resides in `CLAUDE.md`; reviewers confirm it mirrors the latest constitution version.

**Version**: 2.2.1 | **Ratified**: 2025-09-18 | **Last Amended**: 2025-09-18
