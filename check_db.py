import sqlite3, os

# Check which DB path the app uses
db_path = os.environ.get('DB_PATH', '/tmp/learnsphere.db')
print(f"DB_PATH env: {db_path}")

# Check if the file exists
for p in [db_path, 'learnsphere.db', '/tmp/learnsphere.db']:
    exists = os.path.exists(p)
    print(f"  {p} exists: {exists}")
    if exists:
        try:
            conn = sqlite3.connect(p)
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in c.fetchall()]
            print(f"    Tables: {tables}")
            
            if 'users' in tables:
                c.execute("SELECT id, username, email, role FROM users LIMIT 5")
                users = c.fetchall()
                print(f"    Users: {users}")
            conn.close()
        except Exception as e:
            print(f"    Error: {e}")
