# backend/modules/scraper.py
import time
import logging
from apify_client import ApifyClient

logger = logging.getLogger(__name__)


def scrape_tiktok_comments(
    apify_token: str,
    video_url: str,
    max_comments: int = 20,
    max_wait_sec: int = 600,      # 10 menit; bisa diatur via ENV
    poll_interval: float = 3.0,   # detik
    max_retries: int = 2
) -> list:
    """
    Jalankan actor Apify untuk ambil komentar TikTok.
    - Tidak pakai argumen 'wait_for_finish' (tidak didukung di client kamu).
    - Poll status run hingga SUCCEEDED / timeout.
    - Jika gagal atau kosong → kembalikan [] supaya pipeline lanjut aman.
    """
    actor_id = "BDec00yAmCm1QbMEI"
    client = ApifyClient(apify_token)

    for attempt in range(1, max_retries + 2):
        try:
            logger.info(
                f"Starting scraping (attempt {attempt}) for URL: {video_url}...")
            # Panggil tanpa wait_for_finish
            run = client.actor(actor_id).call(run_input={
                "postURLs": [video_url],
                "commentsPerPost": max_comments
            })

            run_id = run.get("id") or run.get("data", {}).get("id")
            if not run_id:
                logger.error("Apify run_id not found from call().")
                continue

            # Poll status sampai selesai / timeout
            start = time.time()
            status = "READY"
            ds_id = None
            while time.time() - start < max_wait_sec:
                r = client.run(run_id).get()
                status = r.get("status", "UNKNOWN")
                ds_id = r.get("defaultDatasetId")
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    break
                time.sleep(poll_interval)

            if status != "SUCCEEDED":
                logger.warning(f"Apify status = {status} (not SUCCEEDED).")
                continue

            # Ambil hasil dataset
            comments = []
            if ds_id:
                for item in client.dataset(ds_id).iterate_items():
                    txt = item.get("text")
                    cid = item.get("commentId")
                    if txt:
                        comments.append({"id": cid, "text": txt})

            logger.info(f"Successfully fetched {len(comments)} comments.")
            return comments[:max_comments]

        except Exception as e:
            logger.error(f"Scrape attempt {attempt} failed: {e}")

    logger.error("All scrape attempts failed/empty, returning [].")
    return []
