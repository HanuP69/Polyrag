import psycopg2

conn = psycopg2.connect("postgresql://postgres:tan69@localhost:5433/polyrag")
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
cur.execute("SELECT extversion FROM pg_extension WHERE extname = %s", ("vector",))
row = cur.fetchone()
print(f"pgvector version: {row[0]}" if row else "pgvector NOT installed")
cur.execute("SELECT version()")
print(f"PostgreSQL: {cur.fetchone()[0]}")
conn.close()
print("pgvector is ready!")
