from __future__ import annotations

import logging
import time
from datetime import datetime

from .backtest_jobs import run_backtest_job
from .models import Backtest, db_session
from .settings import BACKTEST_WORKER_ENABLED, BACKTEST_WORKER_POLL_SECONDS

logger = logging.getLogger(__name__)


def _claim_next_job() -> str | None:
    with db_session() as session:
        job = (
            session.query(Backtest)
            .filter_by(status="pending")
            .order_by(Backtest.created_at.asc())
            .first()
        )
        if not job:
            return None
        job.status = "processing"
        job.progress = job.progress or 0
        job.updated_at = datetime.utcnow()
        session.commit()
        return job.id


def run_worker() -> None:
    logger.info("Backtest worker started.")
    while True:
        if not BACKTEST_WORKER_ENABLED:
            time.sleep(BACKTEST_WORKER_POLL_SECONDS)
            continue
        job_id = _claim_next_job()
        if not job_id:
            time.sleep(BACKTEST_WORKER_POLL_SECONDS)
            continue
        run_backtest_job(job_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_worker()


if __name__ == "__main__":
    main()
