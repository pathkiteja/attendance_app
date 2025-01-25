import sqlite3

DATABASE = 'face_database.db'

def update_schema():
    """Add missing face_encoding column to the database if it doesn't exist."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        # Check if face_encoding column already exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'face_encoding' not in columns:
            # Add the face_encoding column
            cursor.execute("ALTER TABLE users ADD COLUMN face_encoding BLOB")
            print("Database schema updated: 'face_encoding' column added.")
        else:
            print("Database schema is already up-to-date.")

if __name__ == '__main__':
    update_schema()
