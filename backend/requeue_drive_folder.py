"""
One-off: re-queue every non-completed Google Drive document in a KB folder so it
re-runs through the ingestion pipeline (e.g. after an extraction-code change like
the VLM fallback). Reconstructs the per-file Celery task args from the existing
document rows + their stored Drive metadata.

Usage:
    uv run python requeue_drive_folder.py "Solution page brochure"
    uv run python requeue_drive_folder.py "Solution page brochure" --org f9623cfa-...

Run this only with the worker already running the NEW code — it dispatches tasks
to the same broker the worker consumes.
"""
import argparse
import asyncio
import json

from clients.postgres_client import get_postgres_client
from tasks.ingestion_tasks import process_drive_file_task


async def _requeue(folder_name: str, org_id: str | None) -> int:
    pg = get_postgres_client()
    pool = await pg.get_pool()
    async with pool.acquire() as c:
        params = [folder_name]
        org_clause = ""
        if org_id:
            params.append(org_id)
            org_clause = "AND organization_id = $2"
        rows = await c.fetch(
            f"""
            SELECT id, filename, file_key, folder_name, user_id,
                   organization_id, metadata, status
            FROM documents
            WHERE folder_name = $1
              {org_clause}
              AND status <> 'completed'
              AND metadata->>'source' = 'google_drive'
            """,
            *params,
        )

    queued = 0
    for r in rows:
        md = r["metadata"]
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except (ValueError, TypeError):
                md = {}
        md = md or {}
        drive_file_id = md.get("drive_file_id")
        if not drive_file_id:
            print(f"  skip (no drive_file_id): {r['filename']}")
            continue

        # Reset the row back to a clean processing state.
        async with pool.acquire() as c:
            await c.execute(
                """
                UPDATE documents
                SET status='processing', processing_stage='initializing',
                    error=NULL, failed_at=NULL
                WHERE id=$1
                """,
                r["id"],
            )

        process_drive_file_task.delay(
            document_id=str(r["id"]),
            drive_file_id=drive_file_id,
            drive_mime_type=md.get("drive_mime_type"),
            file_name=md.get("drive_file_name") or r["filename"],
            file_key=r["file_key"],
            folder_name=r["folder_name"],
            user_id=str(r["user_id"]),
            organization_id=str(r["organization_id"]),
        )
        queued += 1
        print(f"  re-queued [{r['status']}] {r['filename']}")

    return queued


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_name")
    ap.add_argument("--org", default=None, help="restrict to one organization_id")
    args = ap.parse_args()

    n = asyncio.run(_requeue(args.folder_name, args.org))
    print(f"\n✅ Re-queued {n} document(s) from folder '{args.folder_name}'")


if __name__ == "__main__":
    main()
