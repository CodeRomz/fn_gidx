Gemini Index Sync (GIDX)

Purpose
This tool scans a local directory, generates a searchable index PDF,
and keeps a Gemini File Search Store in sync (uploads, replaces, and removals).
The local directory is the source of truth.

Name
GIDX stands for "Gemini Index Sync".

Quick Start
1) Install dependencies:
   - reportlab (for PDF generation)
   - google-genai (for File Search Store sync)
2) Copy env_template.txt to .env and fill required values.
3) Run:
   python fn_gidx/fn_gidx.py

Environment Variables
Required:
- ROOT_FOLDER
- OUTPUT_DIR
- GEMINI_API_KEY
- FILE_SEARCH_STORE

Optional:
- SKIP_FOLDER
- MAX_FILE_SIZE_MB
- CHECK_PDF_MAGIC
- SCAN_CONCURRENCY
- OUTPUT_PDF
- STATE_DB
- LOG_FILE
- DRY_RUN
- CONFIRM_REMOVALS
- ALLOW_REMOVALS
- NO_PROMPT
- REMOVAL_THRESHOLD
- SAFE_REPLACE
- VERIFY_REMOTE
- MAX_CONCURRENCY

Flow Summary
Step 1:
- Scan ROOT_FOLDER for PDFs (skip rules, size/magic checks).
- Scanning can run in parallel when SCAN_CONCURRENCY > 1 (network shares may benefit).
- Generate gidx_index.pdf (chunked table).
- Update SQLite state DB (includes the index PDF itself).

Step 2 (only if DRY_RUN=0):
- Upload new PDFs and replace changed PDFs.
- Uploads can run in parallel when MAX_CONCURRENCY > 1.
- If a local file is removed and ALLOW_REMOVALS=1, delete its remote doc.
- If removals exceed REMOVAL_THRESHOLD (default 20), prompt before proceeding.
- If SAFE_REPLACE=1, upload first and delete old remote docs after success.
- Upload or replace gidx_index.pdf only when local content changes.

Scheduler Notes (Windows Task Scheduler)
- For unattended runs, set:
  - NO_PROMPT=1
  - CONFIRM_REMOVALS=0 (optional)
  - ALLOW_REMOVALS=1 (only if you want remote deletes)
- In NO_PROMPT mode, removals without a stored remote_document_name are skipped
  to avoid interactive selection prompts.

Safety Tips
- Keep OUTPUT_DIR outside ROOT_FOLDER to avoid accidental indexing loops.
- If you change ROOT_FOLDER or SKIP_FOLDER, review removals before enabling
  ALLOW_REMOVALS.

Operational Notes
- Deleting `OUTPUT_DIR\\gidx_index.pdf` is safe. It is regenerated on the next run
  and uploaded again if it changes or if `VERIFY_REMOTE=1` finds it missing.
- Deleting `OUTPUT_DIR\\gidx_state.sqlite` resets local sync state. The next run
  treats every local PDF as new and uploads them again, which can create duplicate
  remote documents if the store already contains prior uploads. Removals are not
  triggered from an empty state.

Recommended Files to Keep
- `OUTPUT_DIR\\gidx_state.sqlite` preserves upload history and prevents duplicates.
- `OUTPUT_DIR\\gidx_index.pdf` is the generated index artifact; safe to delete but
  useful for local review.
- `OUTPUT_DIR\\gidx_sync.log` helps diagnose sync issues and retries.
