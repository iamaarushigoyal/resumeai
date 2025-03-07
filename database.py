import sqlite3

# --------------------- DATABASE INITIALIZATION --------------------- #
def create_database():
    """Ensure the database and candidates table exist before use."""
    conn = sqlite3.connect("resume_ai.db")  # Connect to SQLite database
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        contact TEXT,
        university TEXT,
        degree TEXT,
        skills TEXT,
        experience TEXT,
        suggested_jobs TEXT,
        uploaded_resume TEXT
    )
    """)
    conn.commit()
    conn.close()


# --------------------- STORE CANDIDATE INFORMATION --------------------- #
def store_candidate_info(name, email, contact, university, degree, skills, experience, suggested_jobs, resume_text):
    """Store candidate details in the database."""
    conn = sqlite3.connect("resume_ai.db")  # Connect to the database
    cursor = conn.cursor()

    # Convert lists to strings for storage
    skills_str = ", ".join(skills)
    experience_str = ", ".join(experience)
    suggested_jobs_str = ", ".join([f"{job} ({confidence}%)" for job, confidence in suggested_jobs])

    try:
        cursor.execute("""
        INSERT INTO candidates (name, email, contact, university, degree, skills, experience, suggested_jobs, uploaded_resume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, email, contact, university, degree, skills_str, experience_str, suggested_jobs_str, resume_text))

        conn.commit()
        print("✅ Candidate information stored successfully!")
    except sqlite3.OperationalError as e:
        print(f"❌ Error inserting candidate info: {e}")
    finally:
        conn.close()


# --------------------- FETCH CANDIDATE DATA --------------------- #
def get_all_candidates():
    """Fetch all candidate records from the database."""
    conn = sqlite3.connect("resume_ai.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates")
    rows = cursor.fetchall()
    conn.close()
    return rows
