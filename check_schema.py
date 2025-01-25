import sqlite3

DATABASE = 'face_database.db'

def check_schema():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users);")
        schema = cursor.fetchall()
        print("Current schema of the 'users' table:")
        for column in schema:
            print(column)

if __name__ == '__main__':
    check_schema()
