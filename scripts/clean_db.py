"""Clean stale test data from SQLite and deduplicate chunks."""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "polyrag.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Remove old test data
cur.execute("""DELETE FROM chunks WHERE content LIKE '%SERVICE AGREEMENT%'
    OR content LIKE '%INDEMNIFICATION%'
    OR content LIKE '%INTELLECTUAL PROPERTY%'
    OR content LIKE '%TERM AND TERMINATION%'""")
print(f"Deleted {cur.rowcount} old test chunks")
conn.commit()

# Deduplicate by content
cur.execute("""DELETE FROM chunks WHERE rowid NOT IN
    (SELECT MIN(rowid) FROM chunks GROUP BY content)""")
print(f"Deduped {cur.rowcount} duplicates")
conn.commit()

# Show what's left
cur.execute("SELECT expert_id, COUNT(*) FROM chunks GROUP BY expert_id")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]} chunks")

cur.execute("SELECT COUNT(*) FROM chunks")
print(f"Total remaining: {cur.fetchone()[0]} chunks")

conn.close()
print("Done.")
