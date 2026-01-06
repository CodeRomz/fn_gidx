from __future__ import annotations

"""Local QMS indexer and Gemini File Search Store sync tool."""

import logging
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from xml.sax.saxutils import escape

try:
    from google import genai
except Exception:  # pragma: no cover - runtime guard if dependency missing
    genai = None

ENV_REFERENCE = Path(__file__).with_name("env_reference.txt")
DOTENV = Path(__file__).with_name(".env")

SCAN_SHOW_EVERY = 5
SKIP_HIDDEN_DIRS = True
FOLLOW_SYMLINKS = False
CHUNK_SIZE = 200


@dataclass(frozen=True)
class DocumentInfo:
    """Lightweight view of a Gemini document listing."""

    name: str
    display_name: str
    mime_type: str
    create_time: str


@dataclass(frozen=True)
class FileRow:
    """Normalized metadata for a scanned PDF file."""

    local_path: Path
    file_name: str
    folder: str
    modified_utc: str
    modified_ns: int
    size_bytes: int


@dataclass(frozen=True)
class SyncAction:
    """Represents the intended sync action for a local file."""

    local_path: Path
    file_name: str
    modified_ns: int
    size_bytes: int
    action: str
    remote_document_name: str


def _load_env_file(path: Path, *, override: bool) -> None:
    """Load key=value lines into environment."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if key in os.environ and not override:
                continue
            os.environ[key] = value


def load_environment() -> None:
    """Load env_reference as defaults, then .env overrides."""
    _load_env_file(ENV_REFERENCE, override=False)
    _load_env_file(DOTENV, override=True)


def get_env(name: str) -> str:
    """Return required env var or raise a friendly error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name}. Set it in {DOTENV.name} or {ENV_REFERENCE.name}.")
    return value


def get_env_optional(name: str) -> str:
    """Return an optional env var (empty string if unset)."""
    return os.getenv(name, "").strip()


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment values with a default."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y"}


def _parse_int_env(name: str) -> Optional[int]:
    """Parse an integer environment value or return None."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _normalize_store(store_name: str) -> str:
    """Normalize a File Search Store name to full resource format."""
    store_name = (store_name or "").strip()
    if store_name.startswith("fileSearchStores/"):
        return store_name
    return f"fileSearchStores/{store_name}"


def _get_attr(obj: object, key: str, default: str = "") -> str:
    """Fetch attribute or dict key from a response object."""
    if isinstance(obj, dict):
        return str(obj.get(key, default) or "")
    return str(getattr(obj, key, default) or "")


def _build_client() -> "genai.Client":
    """Create a Gemini client using GEMINI_API_KEY."""
    if genai is None:
        raise RuntimeError("Missing dependency: install google-genai to manage the file store.")
    api_key = get_env("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def _call_list_documents(client: "genai.Client", store_name: str) -> Iterable:
    """Call the list documents API with signature fallbacks."""
    documents = getattr(client.file_search_stores, "documents", None)
    if documents is not None and hasattr(documents, "list"):
        return documents.list(parent=store_name)

    list_fn = getattr(client.file_search_stores, "list_documents", None)
    if list_fn is None:
        raise RuntimeError("Client missing file_search_stores.documents.list().")

    for kwargs in (
        {"name": store_name},
        {"file_search_store_name": store_name},
    ):
        try:
            return list_fn(**kwargs)
        except TypeError:
            continue

    return list_fn(store_name)


def _iter_documents(client: "genai.Client", store_name: str) -> Iterator[object]:
    """Yield documents from any supported list response shape."""
    response = _call_list_documents(client, store_name)
    if response is None:
        return iter(())
    if hasattr(response, "__iter__"):
        return iter(response)
    if hasattr(response, "documents"):
        return iter(response.documents)
    return iter(())


def find_document(
    client: "genai.Client",
    store_name: str,
    display_name: str,
    *,
    case_sensitive: bool = False,
) -> List[DocumentInfo]:
    """Find documents by exact display name match."""
    store_name = _normalize_store(store_name)
    needle = (display_name or "").strip()
    if not needle:
        raise ValueError("Display name is required to search.")

    if not case_sensitive:
        needle = needle.lower()

    matches: List[DocumentInfo] = []

    for doc in _iter_documents(client, store_name):
        display = _get_attr(doc, "display_name")
        compare = display if case_sensitive else display.lower()
        if compare != needle:
            continue
        matches.append(
            DocumentInfo(
                name=_get_attr(doc, "name"),
                display_name=display,
                mime_type=_get_attr(doc, "mime_type"),
                create_time=_get_attr(doc, "create_time"),
            )
        )

    return matches


def delete_document(client: "genai.Client", document_name: str) -> None:
    """Delete a document by resource name."""
    documents = getattr(client.file_search_stores, "documents", None)
    if documents is not None and hasattr(documents, "delete"):
        try:
            documents.delete(name=document_name, config={"force": True})
        except TypeError:
            documents.delete(name=document_name)
        return

    delete_fn = getattr(client.file_search_stores, "delete_document", None)
    if delete_fn is None:
        raise RuntimeError("Client missing file_search_stores.documents.delete().")

    for kwargs in (
        {"name": document_name, "config": {"force": True}},
        {"document_name": document_name, "config": {"force": True}},
        {"name": document_name},
        {"document_name": document_name},
    ):
        try:
            delete_fn(**kwargs)
            return
        except TypeError:
            continue

    delete_fn(document_name)



def _get_document_by_name(client: "genai.Client", document_name: str) -> Optional[object]:
    """Fetch a document by resource name; return None if missing or unsupported."""
    documents = getattr(client.file_search_stores, "documents", None)
    if documents is not None and hasattr(documents, "get"):
        for kwargs in (
            {"name": document_name},
            {"document_name": document_name},
        ):
            try:
                return documents.get(**kwargs)
            except TypeError:
                continue
            except Exception:
                return None

    get_fn = getattr(client.file_search_stores, "get_document", None)
    if get_fn is None:
        return None

    for kwargs in (
        {"name": document_name},
        {"document_name": document_name},
    ):
        try:
            return get_fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None

    try:
        return get_fn(document_name)
    except Exception:
        return None


def _remote_document_exists(
    client: "genai.Client",
    store_name: str,
    file_name: str,
    remote_document_name: str,
    *,
    allow_fallback: bool,
) -> bool:
    """Check if a remote document exists by name or display name."""
    if remote_document_name:
        doc = _get_document_by_name(client, remote_document_name)
        if doc is not None:
            return True

    if not allow_fallback:
        return False

    matches = find_document(client, store_name, file_name)
    return bool(matches)


def looks_like_pdf(path: Path, *, check_magic: bool) -> bool:
    """Verify PDF magic bytes."""
    if not check_magic:
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"%PDF"
    except Exception:
        return False


def preflight_size_ok(path: Path, max_mb: Optional[int]) -> Tuple[bool, Optional[str]]:
    """Check if file size is within limits."""
    if max_mb is None or max_mb <= 0:
        return True, None
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            return False, f"exceeds {max_mb}MB limit"
        return True, None
    except Exception as exc:
        return False, f"stat error: {exc}"


def parse_skip_folders(raw_value: str, root: Path) -> List[Path]:
    """Parse SKIP_FOLDER env var into absolute paths."""
    if not raw_value:
        return []

    parts = [p.strip() for p in raw_value.replace(",", ";").split(";") if p.strip()]
    skip_dirs: List[Path] = []

    for part in parts:
        p = Path(part)
        if not p.is_absolute():
            p = root / part
        try:
            skip_dirs.append(p.resolve())
        except Exception:
            continue

    return skip_dirs


def is_subpath(path: Path, parent: Path) -> bool:
    """Check if path is under parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def scan_pdfs_with_progress(
    root: Path,
    skip_dirs: List[Path],
    *,
    check_magic: bool,
    max_mb: Optional[int],
) -> List[Path]:
    """Recursively scan for PDFs with live progress."""
    found: List[Path] = []
    visited_entries = 0
    visited_files = 0

    print(f"[SCAN] Starting recursive scan: {root}")
    if skip_dirs:
        print("[SCAN] Skipping folders:")
        for s in skip_dirs:
            print(f"       - {s}")

    start = time.monotonic()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=FOLLOW_SYMLINKS):
        current_dir = Path(dirpath).resolve()

        visited_entries += 1
        if visited_entries % SCAN_SHOW_EVERY == 0:
            elapsed = time.monotonic() - start
            print(
                f"\r[SCAN] Entries={visited_entries} Files={visited_files} Found={len(found)} "
                f"Elapsed={elapsed:.1f}s",
                end="",
                flush=True,
            )

        if any(is_subpath(current_dir, sd) for sd in skip_dirs):
            print(f"\r[SCAN][SKIP] {current_dir} (matches SKIP_FOLDER)" + " " * 20)
            dirnames[:] = []
            continue

        if SKIP_HIDDEN_DIRS:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for fn in filenames:
            visited_entries += 1
            visited_files += 1

            if visited_entries % SCAN_SHOW_EVERY == 0:
                elapsed = time.monotonic() - start
                print(
                    f"\r[SCAN] Entries={visited_entries} Files={visited_files} Found={len(found)} "
                    f"Elapsed={elapsed:.1f}s",
                    end="",
                    flush=True,
                )

            if not fn.lower().endswith(".pdf"):
                continue

            p = Path(dirpath) / fn

            if check_magic and not looks_like_pdf(p, check_magic=check_magic):
                print(f"\r[SCAN][SKIP] {p} (invalid PDF header)" + " " * 20)
                continue

            ok, reason = preflight_size_ok(p, max_mb)
            if not ok:
                print(f"\r[SCAN][SKIP] {p} ({reason})" + " " * 20)
                continue

            found.append(p)

    print()
    elapsed = time.monotonic() - start
    print(f"[SCAN] Complete: {len(found)} PDFs | {visited_files} files scanned | {elapsed:.1f}s")
    return sorted(found)


def _modified_utc_from_ns(modified_ns: int) -> str:
    """Convert nanosecond mtime to an ISO-8601 UTC string."""
    dt = datetime.fromtimestamp(modified_ns / 1e9, tz=timezone.utc)
    return dt.isoformat()


def fetch_local_pdf_rows(root_folder: Path) -> List[FileRow]:
    """Scan the local filesystem and return rows for indexing."""
    skip_raw = get_env_optional("SKIP_FOLDER")
    max_mb = _parse_int_env("MAX_FILE_SIZE_MB")
    check_magic = _parse_bool_env("CHECK_PDF_MAGIC", True)

    skip_dirs = parse_skip_folders(skip_raw, root_folder)
    output_dir_raw = get_env_optional("OUTPUT_DIR")
    if output_dir_raw:
        output_dir = Path(output_dir_raw)
        try:
            output_dir = output_dir.expanduser().resolve(strict=False)
            skip_dirs.append(output_dir)
        except Exception:
            pass
    pdfs = scan_pdfs_with_progress(root_folder, skip_dirs, check_magic=check_magic, max_mb=max_mb)

    rows: List[FileRow] = []
    for pdf in pdfs:
        try:
            stat = pdf.stat()
        except Exception:
            continue
        modified_ns = int(stat.st_mtime_ns)
        rel_folder = ""
        try:
            rel_folder = str(pdf.parent.relative_to(root_folder))
        except Exception:
            rel_folder = pdf.parent.name
        rows.append(
            FileRow(
                local_path=pdf,
                file_name=pdf.name,
                folder=rel_folder or "",
                modified_utc=_modified_utc_from_ns(modified_ns),
                modified_ns=modified_ns,
                size_bytes=int(stat.st_size),
            )
        )
    return rows


def build_index_row(pdf_path: Path) -> Optional[FileRow]:
    """Build a FileRow for the generated index PDF so it can be synced."""
    if not pdf_path.is_file():
        return None
    try:
        stat = pdf_path.stat()
    except Exception:
        return None
    modified_ns = int(stat.st_mtime_ns)
    return FileRow(
        local_path=pdf_path,
        file_name=pdf_path.name,
        folder=str(pdf_path.parent),
        modified_utc=_modified_utc_from_ns(modified_ns),
        modified_ns=modified_ns,
        size_bytes=int(stat.st_size),
    )


def export_gidx_pdf(rows: List[FileRow], output_path: Path) -> Path:
    """Generate a PDF table with file metadata."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, Table, TableStyle
    except Exception:
        raise RuntimeError("Missing dependency: install reportlab to generate PDF.")

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Heading5"],
        textColor=colors.white,
        alignment=1,
        fontSize=9,
        leading=11,
    )
    body_style = ParagraphStyle(
        "TableBody",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    header_row = [
        Paragraph("File Name", header_style),
        Paragraph("Folder", header_style),
        Paragraph("Full Path", header_style),
        Paragraph("Modified UTC", header_style),
        Paragraph("Size (bytes)", header_style),
    ]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=12,
        rightMargin=12,
        topMargin=12,
        bottomMargin=12,
    )
    table_width = doc.width
    col_widths = [
        table_width * 0.20,
        table_width * 0.20,
        table_width * 0.32,
        table_width * 0.16,
        table_width * 0.12,
    ]
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F3B52")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    story = []
    chunk_rows = [header_row]
    for row in rows:
        chunk_rows.append(
            [
                Paragraph(escape(row.file_name), body_style),
                Paragraph(escape(row.folder), body_style),
                Paragraph(escape(str(row.local_path)), body_style),
                Paragraph(escape(row.modified_utc), body_style),
                Paragraph(str(row.size_bytes), body_style),
            ]
        )
        if len(chunk_rows) - 1 >= CHUNK_SIZE:
            table = Table(chunk_rows, colWidths=col_widths, repeatRows=1, splitByRow=1)
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 6))
            chunk_rows = [header_row]

    if len(chunk_rows) > 1:
        table = Table(chunk_rows, colWidths=col_widths, repeatRows=1, splitByRow=1)
        table.setStyle(table_style)
        story.append(table)

    doc.build(story)
    return output_path


def init_db(db_path: Path) -> None:
    """Create the local state database if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                local_path TEXT PRIMARY KEY,
                file_name TEXT,
                modified_ns INTEGER,
                size_bytes INTEGER,
                last_uploaded_ns INTEGER,
                last_uploaded_size INTEGER,
                remote_document_name TEXT,
                last_uploaded_utc TEXT,
                last_error TEXT
            )
            """
        )


def upsert_seen(db_path: Path, rows: List[FileRow]) -> None:
    """Upsert current scan metadata into the state database."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            """
            INSERT INTO files (local_path, file_name, modified_ns, size_bytes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(local_path) DO UPDATE SET
                file_name=excluded.file_name,
                modified_ns=excluded.modified_ns,
                size_bytes=excluded.size_bytes
            """,
            [
                (str(row.local_path), row.file_name, row.modified_ns, row.size_bytes)
                for row in rows
            ],
        )


def load_state(db_path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Load sync state from SQLite into memory."""
    state: Dict[str, Dict[str, Optional[str]]] = {}
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM files"):
            state[str(row["local_path"])] = dict(row)
    return state


def _action_for_row(row: FileRow, record: Optional[Dict[str, Optional[str]]]) -> Optional[SyncAction]:
    """Determine if a file needs upload or replacement."""
    if record is None:
        return SyncAction(
            local_path=row.local_path,
            file_name=row.file_name,
            modified_ns=row.modified_ns,
            size_bytes=row.size_bytes,
            action="upload",
            remote_document_name="",
        )

    last_uploaded_ns = record.get("last_uploaded_ns")
    last_uploaded_size = record.get("last_uploaded_size")
    remote_name = record.get("remote_document_name") or ""

    if last_uploaded_ns is None or last_uploaded_size is None:
        return SyncAction(
            local_path=row.local_path,
            file_name=row.file_name,
            modified_ns=row.modified_ns,
            size_bytes=row.size_bytes,
            action="upload",
            remote_document_name=remote_name,
        )

    if row.modified_ns != int(last_uploaded_ns) or row.size_bytes != int(last_uploaded_size):
        return SyncAction(
            local_path=row.local_path,
            file_name=row.file_name,
            modified_ns=row.modified_ns,
            size_bytes=row.size_bytes,
            action="replace",
            remote_document_name=remote_name,
        )

    return None


def compute_actions(rows: List[FileRow], state: Dict[str, Dict[str, Optional[str]]]) -> List[SyncAction]:
    """Compute upload/replace actions based on last uploaded metadata."""
    actions: List[SyncAction] = []
    for row in rows:
        action = _action_for_row(row, state.get(str(row.local_path)))
        if action:
            actions.append(action)
    return actions


def compute_index_action(
    index_row: Optional[FileRow],
    state: Dict[str, Dict[str, Optional[str]]],
) -> Optional[SyncAction]:
    """Compute sync action for the generated index PDF."""
    if index_row is None:
        return None
    return _action_for_row(index_row, state.get(str(index_row.local_path)))


def update_upload_success(
    db_path: Path,
    row: FileRow,
    document_name: str,
) -> None:
    """Persist successful upload metadata back into the state DB."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            UPDATE files
            SET last_uploaded_ns = ?,
                last_uploaded_size = ?,
                remote_document_name = ?,
                last_uploaded_utc = ?,
                last_error = NULL
            WHERE local_path = ?
            """,
            (
                row.modified_ns,
                row.size_bytes,
                document_name,
                row.modified_utc,
                str(row.local_path),
            ),
        )


def update_last_error(db_path: Path, local_path: Path, message: str) -> None:
    """Persist the last error message for a file."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE files SET last_error = ? WHERE local_path = ?",
            (message, str(local_path)),
        )


def delete_state_entry(db_path: Path, local_path: str) -> None:
    """Remove a file entry from the state database."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM files WHERE local_path = ?", (local_path,))


def wait_for_operation(client: "genai.Client", operation, *, polls: int = 60, delay_sec: int = 2) -> object:
    """Poll a Gemini long-running operation until completion."""
    if not getattr(operation, "done", False):
        for _ in range(polls):
            time.sleep(delay_sec)
            operation = client.operations.get(operation)
            if getattr(operation, "done", False):
                break
    if getattr(operation, "error", None):
        raise RuntimeError(operation.error)
    return operation


def _parse_create_time(raw_value: str) -> datetime:
    """Parse an RFC3339 timestamp into a timezone-aware datetime."""
    value = (raw_value or "").strip()
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _pick_newest_document(matches: List[DocumentInfo]) -> Optional[DocumentInfo]:
    """Pick the most recently created document from a list."""
    if not matches:
        return None
    return max(matches, key=lambda doc: _parse_create_time(doc.create_time))


def upload_pdf_to_gemini(
    client: "genai.Client",
    store_name: str,
    file_path: Path,
) -> str:
    """Upload a PDF to Google File Search Store."""
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    op = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store_name,
        file=str(file_path),
        config={"display_name": file_path.name, "mime_type": "application/pdf"},
    )

    op = wait_for_operation(client, op)

    doc_name = getattr(getattr(op, "response", None), "document_name", None)
    if doc_name:
        return doc_name

    matches = find_document(client, store_name, file_path.name)
    newest = _pick_newest_document(matches)
    if not newest:
        raise RuntimeError("Upload finished, but document name not found.")
    return newest.name


def _should_retry(exc: Exception) -> bool:
    """Heuristic retry check for rate-limited or unavailable responses."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    if code in {429, 503}:
        return True
    text = str(exc).lower()
    tokens = (
        "429",
        "too many requests",
        "resource_exhausted",
        "rate limit",
        "rate-limited",
        "quota",
        "503",
        "service unavailable",
        "unavailable",
        "temporarily unavailable",
    )
    return any(token in text for token in tokens)


def _upload_with_retry(
    client: "genai.Client",
    store_name: str,
    row: FileRow,
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Upload with backoff for retryable errors."""
    for attempt in range(max_attempts):
        try:
            document_name = upload_pdf_to_gemini(client, store_name, row.local_path)
            if not document_name:
                return None, "Upload failed without a document name."
            return document_name, None
        except Exception as exc:
            if attempt >= max_attempts - 1 or not _should_retry(exc):
                return None, str(exc)
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay += delay * 0.1 * random.random()
            print(f"[RETRY] upload failed for {row.file_name}: {exc} (retry in {delay:.1f}s)")
            time.sleep(delay)
    return None, "Upload failed without a document name."


def _upload_worker(store_name: str, row: FileRow) -> Tuple[Optional[str], Optional[str]]:
    """Upload a PDF with retry; return (document_name, error_message)."""
    try:
        client = _build_client()
        return _upload_with_retry(client, store_name, row)
    except Exception as exc:
        return None, str(exc)



def _choose_document(matches: List[DocumentInfo]) -> Optional[DocumentInfo]:
    """Prompt the user to choose a document when multiple matches exist."""
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    print("Multiple matches found:")
    for idx, doc in enumerate(matches, start=1):
        print(f"{idx}. {doc.display_name} | {doc.name}")
    selection = input("Choose number to delete: ").strip()
    try:
        idx = int(selection)
        return matches[idx - 1]
    except Exception:
        print("Invalid selection.")
        return None


def delete_remote_document(
    client: "genai.Client",
    store_name: str,
    file_name: str,
    remote_document_name: str,
) -> str:
    """Delete by stored remote name or fallback to display_name search."""
    if remote_document_name:
        delete_document(client, remote_document_name)
        return remote_document_name

    matches = find_document(client, store_name, file_name)
    target = _choose_document(matches)
    if target is None:
        raise RuntimeError(f"No document selected for {file_name}")
    delete_document(client, target.name)
    return target.name


def build_output_paths(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Resolve output paths for PDF, SQLite state, and log file."""
    pdf_path = Path(get_env_optional("OUTPUT_PDF") or (output_dir / "gidx_index.pdf"))
    db_path = Path(get_env_optional("STATE_DB") or (output_dir / "gidx_state.sqlite"))
    log_path = Path(get_env_optional("LOG_FILE") or (output_dir / "gidx_sync.log"))
    return pdf_path, db_path, log_path


def configure_logging(log_path: Path) -> None:
    """Configure file logging for sync runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        filemode="a",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )


def _row_lookup(rows: List[FileRow]) -> Dict[str, FileRow]:
    """Map local_path string to FileRow for quick lookup."""
    return {str(row.local_path): row for row in rows}


def compute_removed_entries(
    state: Dict[str, Dict[str, Optional[str]]],
    current_paths: Iterable[str],
) -> List[Dict[str, Optional[str]]]:
    """Return state entries whose local_path is no longer present."""
    current = set(current_paths)
    removed: List[Dict[str, Optional[str]]] = []
    for local_path, record in state.items():
        if local_path not in current:
            removed.append(record)
    return removed


def main() -> None:
    """Entry point for the local index + sync workflow."""
    load_environment()

    root_raw = get_env("ROOT_FOLDER")
    output_dir = Path(get_env("OUTPUT_DIR"))
    pdf_path, db_path, log_path = build_output_paths(output_dir)

    configure_logging(log_path)

    root_folder = Path(root_raw).expanduser().resolve()
    if not root_folder.exists() or not root_folder.is_dir():
        raise ValueError(f"ROOT_FOLDER is not a valid directory: {root_folder}")

    print("Step 1: scan PDFs and generate index")
    rows = fetch_local_pdf_rows(root_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_gidx_pdf(rows, pdf_path)
    index_row = build_index_row(pdf_path)

    init_db(db_path)
    seen_rows = rows + ([index_row] if index_row else [])
    upsert_seen(db_path, seen_rows)

    print(f"Step 1 done: saved PDF to {pdf_path}")

    dry_run = _parse_bool_env("DRY_RUN", False)
    if dry_run:
        print("Step 2 skipped: DRY_RUN=1")
        return

    print("Step 2: sync local PDFs to Gemini File Search Store")
    store_name = _normalize_store(get_env("FILE_SEARCH_STORE"))
    client = _build_client()

    state = load_state(db_path)
    actions = compute_actions(rows, state)
    row_lookup = _row_lookup(rows)
    index_action = compute_index_action(index_row, state)
    current_paths = list(row_lookup.keys())
    if index_row:
        current_paths.append(str(index_row.local_path))
    removed_entries = compute_removed_entries(state, current_paths)
    has_changes = bool(actions or removed_entries)
    if index_action and index_action.action != "upload" and not has_changes:
        index_action = None

    confirm_removals = _parse_bool_env("CONFIRM_REMOVALS", True)
    allow_removals = _parse_bool_env("ALLOW_REMOVALS", False)
    no_prompt = _parse_bool_env("NO_PROMPT", False)
    safe_replace = _parse_bool_env("SAFE_REPLACE", False)
    verify_remote = _parse_bool_env("VERIFY_REMOTE", False)
    max_concurrency_raw = _parse_int_env("MAX_CONCURRENCY")
    max_concurrency = max(1, max_concurrency_raw or 1)

    if verify_remote:
        print("Remote check enabled: verifying stored documents.")
        action_paths = {str(action.local_path) for action in actions}
        for row in rows:
            path = str(row.local_path)
            if path in action_paths:
                continue
            record = state.get(path)
            if not record:
                continue
            remote_name = record.get("remote_document_name") or ""
            exists = _remote_document_exists(
                client,
                store_name,
                row.file_name,
                remote_name,
                allow_fallback=not bool(remote_name),
            )
            if not exists:
                actions.append(
                    SyncAction(
                        local_path=row.local_path,
                        file_name=row.file_name,
                        modified_ns=row.modified_ns,
                        size_bytes=row.size_bytes,
                        action="upload",
                        remote_document_name=remote_name,
                    )
                )
                action_paths.add(path)

        if index_row and index_action is None:
            record = state.get(str(index_row.local_path))
            remote_name = record.get("remote_document_name") if record else ""
            exists = _remote_document_exists(
                client,
                store_name,
                index_row.file_name,
                remote_name or "",
                allow_fallback=not bool(remote_name),
            )
            if not exists:
                index_action = SyncAction(
                    local_path=index_row.local_path,
                    file_name=index_row.file_name,
                    modified_ns=index_row.modified_ns,
                    size_bytes=index_row.size_bytes,
                    action="upload",
                    remote_document_name=remote_name or "",
                )

    total_found = len(rows)
    new_files = sum(1 for action in actions if action.action == "upload")
    changed_files = sum(1 for action in actions if action.action == "replace")
    unchanged_files = total_found - new_files - changed_files
    removed_files = len(removed_entries)

    print(f"Total found: {total_found}")
    print(f"New files: {new_files}")
    print(f"Changed files: {changed_files}")
    print(f"Unchanged files: {unchanged_files}")
    print(f"Removed files: {removed_files}")
    if index_action:
        print(f"Index PDF: {index_action.action}")
    else:
        print("Index PDF: unchanged")

    if no_prompt:
        proceed = "y"
    else:
        proceed = input("Proceed with sync? [y/N]: ").strip().lower()
    if proceed not in {"y", "yes"}:
        print("Step 2 cancelled by user.")
        return

    if max_concurrency <= 1:
        for action in actions:
            row = row_lookup.get(str(action.local_path))
            if not row:
                continue
            try:
                if action.action == "replace" and not safe_replace:
                    print(f"[REPLACE] {row.file_name} -> deleting old document")
                    delete_remote_document(client, store_name, row.file_name, action.remote_document_name)
                elif action.action == "replace" and safe_replace:
                    print(f"[REPLACE SAFE] {row.file_name} -> upload then delete old document")
                elif action.action == "upload":
                    print(f"[UPLOAD] {row.file_name}")

                document_name, error = _upload_with_retry(client, store_name, row)
                if error:
                    raise RuntimeError(error)

                update_upload_success(db_path, row, document_name)
                print(f"[DONE] {row.file_name} -> {document_name}")

                if action.action == "replace" and safe_replace:
                    if no_prompt and not action.remote_document_name:
                        logging.warning(
                            "Safe replace cleanup skipped for %s: missing remote_document_name in NO_PROMPT mode.",
                            row.file_name,
                        )
                        print(f"[SKIP] {row.file_name}: missing remote_document_name for safe cleanup")
                    else:
                        try:
                            delete_remote_document(
                                client,
                                store_name,
                                row.file_name,
                                action.remote_document_name,
                            )
                        except Exception as exc:
                            logging.warning(
                                "Safe replace cleanup failed for %s: %s",
                                row.file_name,
                                exc,
                            )
                            print(f"[WARN] cleanup failed for {row.file_name}: {exc}")
            except Exception as exc:
                message = str(exc)
                update_last_error(db_path, row.local_path, message)
                logging.error("Sync failed for %s: %s", row.file_name, message)
                print(f"[ERROR] {row.file_name}: {message}")
    else:
        print(f"Upload concurrency: {max_concurrency}")
        pending: List[Tuple[SyncAction, FileRow]] = []
        for action in actions:
            row = row_lookup.get(str(action.local_path))
            if not row:
                continue
            try:
                if action.action == "replace" and not safe_replace:
                    print(f"[REPLACE] {row.file_name} -> deleting old document")
                    delete_remote_document(client, store_name, row.file_name, action.remote_document_name)
                elif action.action == "replace" and safe_replace:
                    print(f"[REPLACE SAFE] {row.file_name} -> upload then delete old document")
                elif action.action == "upload":
                    print(f"[UPLOAD] {row.file_name}")
                pending.append((action, row))
            except Exception as exc:
                message = str(exc)
                update_last_error(db_path, row.local_path, message)
                logging.error("Sync failed for %s: %s", row.file_name, message)
                print(f"[ERROR] {row.file_name}: {message}")

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_map = {
                executor.submit(_upload_worker, store_name, row): (action, row)
                for action, row in pending
            }
            for future in as_completed(future_map):
                action, row = future_map[future]
                try:
                    document_name, error = future.result()
                except Exception as exc:
                    document_name, error = None, str(exc)
                if error:
                    update_last_error(db_path, row.local_path, error)
                    logging.error("Sync failed for %s: %s", row.file_name, error)
                    print(f"[ERROR] {row.file_name}: {error}")
                    continue

                update_upload_success(db_path, row, document_name)
                print(f"[DONE] {row.file_name} -> {document_name}")

                if action.action == "replace" and safe_replace:
                    if no_prompt and not action.remote_document_name:
                        logging.warning(
                            "Safe replace cleanup skipped for %s: missing remote_document_name in NO_PROMPT mode.",
                            row.file_name,
                        )
                        print(f"[SKIP] {row.file_name}: missing remote_document_name for safe cleanup")
                    else:
                        try:
                            delete_remote_document(
                                client,
                                store_name,
                                row.file_name,
                                action.remote_document_name,
                            )
                        except Exception as exc:
                            logging.warning(
                                "Safe replace cleanup failed for %s: %s",
                                row.file_name,
                                exc,
                            )
                            print(f"[WARN] cleanup failed for {row.file_name}: {exc}")

    removal_threshold_raw = _parse_int_env("REMOVAL_THRESHOLD")
    removal_threshold = removal_threshold_raw if removal_threshold_raw is not None else 20

    if removed_entries and not allow_removals:
        removed_entries = []
        print("Removal sync skipped: set ALLOW_REMOVALS=1 to enable.")

    if removed_entries and len(removed_entries) >= removal_threshold:
        if no_prompt:
            removed_entries = []
            logging.warning("Removal sync skipped: threshold exceeded in NO_PROMPT mode.")
            print("Removal sync skipped: removal threshold exceeded in NO_PROMPT mode.")
        else:
            msg = f"Removal count {len(removed_entries)} exceeds threshold {removal_threshold}. Proceed? [y/N]: "
            proceed_threshold = input(msg).strip().lower()
            if proceed_threshold not in {"y", "yes"}:
                removed_entries = []

    if removed_entries and confirm_removals:
        if no_prompt:
            proceed_removals = "y"
        else:
            proceed_removals = input("Proceed with removal sync? [y/N]: ").strip().lower()
        if proceed_removals not in {"y", "yes"}:
            removed_entries = []

    for record in removed_entries:
        local_path = record.get("local_path") or ""
        file_name = record.get("file_name") or ""
        remote_name = record.get("remote_document_name") or ""
        if not file_name:
            continue
        try:
            if no_prompt and not remote_name:
                logging.warning(
                    "Removal skipped for %s: missing remote_document_name in NO_PROMPT mode.",
                    file_name,
                )
                print(f"[SKIP] {file_name}: missing remote_document_name for removal")
                continue
            print(f"[REMOVE] {file_name} -> deleting remote document")
            delete_remote_document(client, store_name, file_name, remote_name)
            delete_state_entry(db_path, local_path)
            print(f"[REMOVE DONE] {file_name}")
        except Exception as exc:
            message = str(exc)
            if local_path:
                update_last_error(db_path, Path(local_path), message)
            logging.error("Remove failed for %s: %s", file_name, message)
            print(f"[REMOVE ERROR] {file_name}: {message}")

    if index_action and index_row:
        try:
            if index_action.action == "replace" and not safe_replace:
                print(f"[INDEX REPLACE] {index_row.file_name} -> deleting old document")
                delete_remote_document(
                    client,
                    store_name,
                    index_row.file_name,
                    index_action.remote_document_name,
                )
            elif index_action.action == "replace" and safe_replace:
                print(f"[INDEX REPLACE SAFE] {index_row.file_name} -> upload then delete old document")
            elif index_action.action == "upload":
                print(f"[INDEX UPLOAD] {index_row.file_name}")

            document_name, error = _upload_with_retry(client, store_name, index_row)
            if error:
                raise RuntimeError(error)

            update_upload_success(db_path, index_row, document_name)
            print(f"[INDEX DONE] {index_row.file_name} -> {document_name}")

            if index_action.action == "replace" and safe_replace:
                if no_prompt and not index_action.remote_document_name:
                    logging.warning(
                        "Index safe replace cleanup skipped: missing remote_document_name in NO_PROMPT mode."
                    )
                    print("[INDEX SKIP] missing remote_document_name for safe cleanup")
                else:
                    try:
                        delete_remote_document(
                            client,
                            store_name,
                            index_row.file_name,
                            index_action.remote_document_name,
                        )
                    except Exception as exc:
                        logging.warning("Index safe replace cleanup failed: %s", exc)
                        print(f"[INDEX WARN] cleanup failed: {exc}")
        except Exception as exc:
            message = str(exc)
            update_last_error(db_path, index_row.local_path, message)
            logging.error("Index sync failed for %s: %s", index_row.file_name, message)
            print(f"[INDEX ERROR] {index_row.file_name}: {message}")

    print("Step 2 done: sync complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
