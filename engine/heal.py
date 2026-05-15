import json
from typing import Optional
from engine.config import DATA_DIR


def get_pipeline_health(org_id: str = None) -> dict:
    from engine.db import get_conn, TESTING

    conn = get_conn()
    health = {
        "total_queries": 0,
        "avg_latency_ms": 0,
        "feedback_count": 0,
        "positive_feedback": 0,
        "negative_feedback": 0,
        "needs_retrain": False,
        "gate_accuracy_estimate": 0.0,
    }

    try:
        if TESTING:
            row = conn.execute(
                "SELECT COUNT(*) as cnt, AVG(latency_ms) as avg_lat FROM query_logs"
                + (" WHERE org_id = ?" if org_id else ""),
                (org_id,) if org_id else ()
            ).fetchone()
            health["total_queries"] = row["cnt"] or 0
            health["avg_latency_ms"] = round(row["avg_lat"] or 0, 1)

            fb_row = conn.execute(
                "SELECT COUNT(*) as cnt, SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as pos, "
                "SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as neg FROM user_feedback"
            ).fetchone()
            health["feedback_count"] = fb_row["cnt"] or 0
            health["positive_feedback"] = fb_row["pos"] or 0
            health["negative_feedback"] = fb_row["neg"] or 0
        else:
            cur = conn.cursor()
            query = "SELECT COUNT(*), AVG(latency_ms) FROM query_logs"
            if org_id:
                query += " WHERE org_id = %s"
                cur.execute(query, (org_id,))
            else:
                cur.execute(query)
            row = cur.fetchone()
            health["total_queries"] = row[0] or 0
            health["avg_latency_ms"] = round(row[1] or 0, 1)

            cur.execute(
                "SELECT COUNT(*), SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) FROM user_feedback"
            )
            fb_row = cur.fetchone()
            health["feedback_count"] = fb_row[0] or 0
            health["positive_feedback"] = fb_row[1] or 0
            health["negative_feedback"] = fb_row[2] or 0

        if health["feedback_count"] > 0:
            health["gate_accuracy_estimate"] = round(
                health["positive_feedback"] / health["feedback_count"], 3
            )

        health["needs_retrain"] = (
            health["negative_feedback"] >= 10
            or (health["feedback_count"] >= 20 and health["gate_accuracy_estimate"] < 0.7)
        )

    except Exception as e:
        print(f"[Heal] Health check error: {e}")

    return health


def should_retrain_gate() -> bool:
    health = get_pipeline_health()
    return health["needs_retrain"]


def get_feedback_for_retraining() -> list[dict]:
    from engine.db import get_conn, TESTING

    conn = get_conn()
    entries = []

    try:
        if TESTING:
            rows = conn.execute(
                """SELECT ql.query, uf.correct_expert, uf.rating
                   FROM user_feedback uf
                   JOIN query_logs ql ON uf.query_log_id = ql.log_id
                   WHERE uf.correct_expert IS NOT NULL"""
            ).fetchall()
            for row in rows:
                entries.append({
                    "query": row["query"],
                    "correct_expert": row["correct_expert"],
                    "rating": row["rating"],
                })
        else:
            cur = conn.cursor()
            cur.execute(
                """SELECT ql.query, uf.correct_expert, uf.rating
                   FROM user_feedback uf
                   JOIN query_logs ql ON uf.query_log_id = ql.log_id
                   WHERE uf.correct_expert IS NOT NULL"""
            )
            for row in cur.fetchall():
                entries.append({
                    "query": row[0],
                    "correct_expert": row[1],
                    "rating": row[2],
                })
    except Exception as e:
        print(f"[Heal] Failed to fetch feedback: {e}")

    return entries
