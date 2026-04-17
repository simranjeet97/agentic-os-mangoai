import asyncio
import aiosqlite
import uuid
import os
from pathlib import Path

async def debug_fts():
    db_path = "debug_episodic.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    from memory.episodic_memory import EpisodicMemory
    em = EpisodicMemory(db_path=db_path)
    await em._ensure_init()
    
    # Insert 
    await em.record(content="Configured docker containers for deployment")
    
    async with aiosqlite.connect(db_path) as db:
        # Check main table
        async with db.execute("SELECT * FROM episodes") as cursor:
            rows = await cursor.fetchall()
            print(f"Main table rows: {len(rows)}")
            
        # Check FTS table
        async with db.execute("SELECT rowid, * FROM episodes_fts") as cursor:
            fts_rows = await cursor.fetchall()
            print(f"FTS table rows: {len(fts_rows)}")
            for r in fts_rows:
                print(dict(r))
                
        # Try search
        query = '"docker"'
        sql = "SELECT * FROM episodes_fts WHERE episodes_fts MATCH ?"
        async with db.execute(sql, [query]) as cursor:
            results = await cursor.fetchall()
            print(f"Search results for {query}: {len(results)}")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    asyncio.run(debug_fts())
