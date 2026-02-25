"""Initialize alert columns if they don't exist"""
from sqlalchemy import text
from app.db import SessionLocal

def ensure_alert_columns():
    """Add alert columns to watchlist table if missing"""
    db = SessionLocal()
    try:
        # Check if alert_upper column exists
        result = db.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='watchlist' AND column_name='alert_upper'
            )
        """))
        
        if not result.scalar():
            print("Adding alert columns to watchlist...")
            db.execute(text("ALTER TABLE watchlist ADD COLUMN alert_upper NUMERIC(20, 6)"))
            db.execute(text("ALTER TABLE watchlist ADD COLUMN alert_lower NUMERIC(20, 6)"))
            db.execute(text("ALTER TABLE watchlist ADD COLUMN alert_triggered BOOLEAN DEFAULT false"))
            db.commit()
            print("✅ Alert columns added successfully")
        else:
            print("✅ Alert columns already exist")
    except Exception as e:
        print(f"⚠️ Could not add alert columns: {e}")
    finally:
        db.close()
