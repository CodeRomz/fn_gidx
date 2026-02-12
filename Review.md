PROJECT_NAME: `{{fn_gidx}}`
REPO_ROOT: `{{https://github.com/CodeRomz/fn_gidx.git}}`
REVIEW_MODE: `deep`  # options: quick | standard | deep
ALLOW_EXTERNAL_SOURCES: `false`  # if true: you may consult official docs ONLY for best-practice comparisons, not for guessing behavior
RUNTIME_TARGET: `{{Python 3.12 on Windows 11 Pro}}`  # ex: Python 3.11 on AlmaLinux 9

SCOPE & SOURCE OF TRUTH

* Treat this repository as the source of truth for “what the software does.”
* Do not invent features or behaviors that are not explicitly present in code/config/tests.
* You MAY use general Python engineering knowledge to evaluate quality, security, performance, and maintainability.
* If something cannot be proven from the repo, mark it as Unknown and list what evidence is missing (file, config, env var, secret, sample data, etc.).

PROJECT DETECTION

* Scan all files (recursively) and review them line by line (prioritize core runtime code first).
* Identify and confirm:

  * Project type: library, CLI tool, web API/service, background worker, automation script, data pipeline, notebook-heavy repo, etc.
  * Primary entrypoint(s): `__main__.py`, console scripts, `main.py`, `app.py`, `wsgi/asgi`, `uvicorn/gunicorn` target, `click/typer/argparse` command tree, cron/systemd runner, etc.
  * Runtime assumptions: OS, Python version hints, required env vars, filesystem layout, network dependencies, privileged operations.
  * Dependency management: `pyproject.toml` / `poetry.lock` / `requirements*.txt` / `Pipfile*` / `setup.cfg` / `setup.py`.
  * Execution flows: “how it starts” and “what it does” from startup to shutdown.

FILE CLASSIFICATION

* For every file, classify it by role (use the closest match; add a new bucket if needed):

  * core_package: Python package code (business logic, domain layer)
  * cli: command-line interfaces and command trees
  * api_service: web app / routes / handlers / middleware
  * workers_jobs: schedulers, queues, cron-like jobs, background tasks
  * integrations: external APIs/SDKs, DB clients, message brokers, third-party connectors
  * persistence: DB models/migrations, repositories/DAOs, schema, ORM usage
  * config: settings loaders, `.env` patterns, config schemas, feature flags
  * security: authn/authz, crypto, secrets handling, permissions, input validation, sandboxing
  * utils_shared: helpers, common libs, constants
  * data_assets: sample data, fixtures, templates, static files
  * tests: unit/integration/e2e tests and what they cover
  * tooling: linters/formatters, pre-commit, scripts, Makefile, tasks runners
  * docs: README, ADRs, design docs, runbooks
  * infra: Docker, Compose, Kubernetes, CI/CD pipelines, systemd, deployment manifests
* For each file, describe its purpose in one short sentence.

LINE-BY-LINE REVIEW (FILE-BY-FILE)
For each significant module/file:

* Explain what it does, its inputs/outputs, and its side effects (filesystem, network, subprocess, DB, global state).
* Identify:

  * Key classes/functions, their responsibilities, and coupling boundaries
  * Data structures and schemas (DTOs, Pydantic models, dataclasses, dict contracts)
  * Error-handling strategy (including whether try/except/else/finally is used appropriately)
  * Logging strategy (what gets logged, potential sensitive data in logs)
  * Configuration access patterns (env vars, config files, defaults, validation)
  * Concurrency model: threads, asyncio, multiprocessing, task queues; any race conditions
  * Performance hot paths: loops, N+1 IO, repeated parsing, heavy imports, blocking calls in async
* Track all “critical paths” (happy path + failure modes) end-to-end.

ARCHITECTURE RECONSTRUCTION
Rebuild a mental model of the system as implemented (not as intended):

* Components: modules/services/layers and their responsibilities
* Data flow: where data comes from, transforms, and ends up (diagram in text is fine)
* Trust boundaries: user input → validation → processing → storage/output
* Integration map: external services, DBs, message queues, file stores, auth providers
* Deployment/runtime: how it runs in prod (process model, ports, workers, env vars, volumes)

SECURITY ENHANCEMENTS (MANDATORY SECTION)

* Identify and prioritize risks (with evidence from code):

  * Secrets exposure (hardcoded tokens, committed keys, weak `.env` patterns)
  * Injection risks (SQL, command, template, JSONPath/JQ-like, LDAP, SSRF, path traversal)
  * Unsafe deserialization (pickle/yaml load), dynamic `eval/exec`, insecure regex usage
  * Auth/authz gaps (missing checks, insecure defaults, privilege escalation)
  * Crypto misuse (weak algorithms, wrong modes, missing randomness, DIY crypto)
  * Supply-chain risks (un-pinned deps, risky packages, download-and-exec patterns)
  * Logging/telemetry leaks (PII/secrets in logs)
* Provide concrete mitigations and “secure-by-default” refactors.

QUALITY & RISK ASSESSMENT
Highlight:

* Design patterns and architectural decisions (good and bad)
* Maintainability: readability, modularity, naming, duplication, complexity, typing
* Testing maturity: coverage hints, test quality, missing tests for critical paths
* Packaging maturity: versioning, semantic release, build backend, license, SBOM readiness
* Operational readiness: configuration validation, health checks, graceful shutdown, observability
* Backwards compatibility and extension stability: what can safely change vs what is a contract
  Clearly separate:
* Known (proven by code/config/tests)
* Unknown (not present, ambiguous, or environment-dependent)

DELIVERABLES FORMAT

* Start with a high-level “What this project is” in 2–4 sentences.
* Then provide:

  1. File inventory summary (bucketed), plus top 10 most important files (table format).
  2. Architecture reconstruction (components + data flow).
  3. Security Enhancements (prioritized findings + fixes).
  4. Quality & maintainability findings (prioritized).
  5. Performance considerations (where relevant).
  6. Safe extension points (where new features can hook in without breaking behavior).
* For lists (findings, risks, TODOs), present them as tables with up to 10 entries by default.

OPTIONAL: LATEST TOOLS SUMMARY (ONLY IF ALLOW_EXTERNAL_SOURCES=true)

* If permitted, briefly mention modern, widely adopted tooling that fits the repo’s style (e.g., ruff, mypy/pyright, pytest, pip-tools/poetry, bandit/semgrep), but do not change conclusions about behavior unless the repo itself supports it.

FINAL SUMMARY
Finish with a concise summary of:

* Robustness and stability
* Maintainability and clarity
* Performance considerations
* Vulnerability posture
* Top 3 recommended next actions (most leverage, least risk)