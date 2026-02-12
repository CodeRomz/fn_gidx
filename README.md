Gemini/Vertex Index Sync (GIDX)

Purpose
This tool scans a local directory, generates a searchable index PDF, and syncs PDFs to a remote document store.
The remote backend can be either:
- Gemini Developer API File Search Store
- Vertex AI RAG Engine Corpus

The local directory is the source of truth.

Name
GIDX stands for "Generative Index Sync".

Quick Start
1) Install dependencies:
   - `reportlab` (PDF generation)
   - `google-genai` (Gemini File Search backend)
   - `google-cloud-aiplatform` (Vertex RAG backend)
2) Copy `env_template.txt` to `.env` and fill values.
3) Run:
   - `python fn_gidx.py`

Backend Selection
`SYNC_BACKEND` is required:
- `gemini_file_search`
- `vertex_rag`

Environment Variables
Required for all runs:
- `ROOT_FOLDER`
- `OUTPUT_DIR`

Backend required (Gemini):
- `GEMINI_API_KEY`
- `FILE_SEARCH_STORE`

Backend required (Vertex):
- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION`
- `VERTEX_RAG_CORPUS` (full resource name recommended)

Optional backend controls:
- `VERTEX_INTERACTIVE_LOGIN` (`1` default, only used for Vertex)

Optional scan/output/sync controls:
- `ROOT_DISPLAY_PATH`
- `SKIP_FOLDER`
- `MAX_FILE_SIZE_MB`
- `CHECK_PDF_MAGIC`
- `SCAN_CONCURRENCY`
- `SCAN_SPINNER`
- `SCAN_THREAD_LINES`
- `PDF_FAST`
- `PDF_BUILD_SPINNER`
- `OUTPUT_PDF`
- `STATE_DB`
- `LOG_FILE`
- `DRY_RUN`
- `CONFIRM_REMOVALS`
- `ALLOW_REMOVALS`
- `NO_PROMPT`
- `REMOVAL_THRESHOLD`
- `SAFE_REPLACE`
- `VERIFY_REMOTE`
- `UPLOAD_CONCURRENCY`
- `PERF_TIMING`
- `PROFILE_RUN`

Vertex Authentication (ADC)
Vertex RAG backend uses Application Default Credentials (ADC).

Recommended setup:
1) `gcloud auth application-default login`
2) `gcloud auth application-default set-quota-project <VERTEX_PROJECT_ID>`

Interactive login flow in script:
- If backend is `vertex_rag` and ADC is missing/expired:
  - with `NO_PROMPT=0` and `VERTEX_INTERACTIVE_LOGIN=1`, the script prompts to run `gcloud` login.
  - with `NO_PROMPT=1` (scheduler mode), the script fails fast and tells you how to configure ADC.

Flow Summary
Step 0:
- Resolve `SYNC_BACKEND` at startup.
- If backend is Vertex, validate ADC before scanning.

Step 1:
- Scan `ROOT_FOLDER` for PDFs (skip rules, size/magic checks).
- Generate `gidx_index.pdf` (chunked table).
- Update SQLite state DB (includes the index PDF itself).

Step 2 (only if `DRY_RUN=0`):
- Upload new PDFs and replace changed PDFs.
- Uploads can run in parallel when `UPLOAD_CONCURRENCY > 1`.
- If a local file is removed and `ALLOW_REMOVALS=1`, delete its remote file/document.
- If removals exceed `REMOVAL_THRESHOLD` (default 20), prompt before proceeding.
- If `SAFE_REPLACE=1`, upload first and delete old remote docs/files after success.
- Upload or replace `gidx_index.pdf` only when local content changes.

Scheduler Notes (Windows Task Scheduler)
- For unattended runs, set:
  - `NO_PROMPT=1`
  - `CONFIRM_REMOVALS=0` (optional)
  - `ALLOW_REMOVALS=1` (only if you want remote deletes)
- Vertex backend in scheduler mode requires ADC preconfigured on the runtime identity.
- In `NO_PROMPT` mode, removals without a stored `remote_document_name` are skipped to avoid interactive selection prompts.

Safety Tips
- Keep `OUTPUT_DIR` outside `ROOT_FOLDER` to avoid accidental indexing loops.
- If you change `ROOT_FOLDER` or `SKIP_FOLDER`, review removals before enabling `ALLOW_REMOVALS`.

Operational Notes
- Deleting `OUTPUT_DIR\\gidx_index.pdf` is safe. It is regenerated on next run and uploaded again if changed or if `VERIFY_REMOTE=1` finds it missing.
- Deleting `OUTPUT_DIR\\gidx_state.sqlite` resets local sync state. Next run treats local PDFs as new and uploads them again, which can create duplicates in the remote backend.

Recommended Files to Keep
- `OUTPUT_DIR\\gidx_state.sqlite` preserves upload history and prevents duplicates.
- `OUTPUT_DIR\\gidx_index.pdf` is the generated index artifact; safe to delete but useful for local review.
- `OUTPUT_DIR\\gidx_sync.log` helps diagnose sync issues and retries.
