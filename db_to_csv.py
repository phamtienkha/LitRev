import sqlite3
import csv

def convert_db_to_csv(db_file, csv_file):
    """Converts a SQLite database to a CSV file.

    Args:
        db_file (str): Path to the .db file.
        csv_file (str): Path to the output .csv file.
    """

    try:
        # Connect to the database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Write each table to the CSV file
        with open(csv_file, 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            for table_name in tables:
                table_name = table_name[0]
                cursor.execute('SELECT * FROM {}'.format(table_name))

                # Write column headers
                writer.writerow([column[0] for column in cursor.description])  

                # Write data rows
                writer.writerows(cursor.fetchall())

    except sqlite3.Error as e:
        print("Error:", e)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    database_file = './data/papers.db'
    output_csv_file = './data/papers.csv'
    convert_db_to_csv(database_file, output_csv_file)