import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, text, Boolean
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from datetime import datetime, timezone

# Pulls from the environment variable constructed in docker-compose.yml
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set!")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Base(DeclarativeBase):
    pass

# --- Table Definitions ---

class EnergySource(Base):
    __tablename__ = "energy_sources"
    id = Column(Integer, primary_key=True)
    source_type = Column(String, unique = True, nullable = False)  # Solar, Wind, Mix, etc.

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    energy_source_id = Column(Integer, ForeignKey("energy_sources.id"))
    prediction_source = Column(String)
    input_date = Column(DateTime)
    input_time_start = Column(Integer)
    input_time_end = Column(Integer)
    predict_result = Column(Float)
    predict_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ml_model = Column(String)

class DataQualityStat(Base):
    __tablename__ = "data_quality_stats"
    
    # The Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # File name column
    file_name = Column(String, index=True)
    
    # Time-Series Tracking column
    ingestion_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # additional info about the bad dataset
    total_rows = Column(Integer)
    error_count = Column(Integer)
    error_rate = Column(Float)
    is_schema_valid = Column(Boolean)
    error_criticality = Column(String)

# --- Database Operations ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception:
        return False

def save_predictions_batch(db: Session, records_data: list):
    try:
        db_records = [PredictionRecord(**data) for data in records_data]
        db.add_all(db_records)
        db.commit()
    except:
        db.rollback()
        raise


def query_predictions(db: Session, ml_model=None, prediction_source=None,energy_source_id=None, start_date=None, end_date=None, limit=100):
    query = db.query(PredictionRecord)
    if ml_model is not None:
        query = query.filter(PredictionRecord.ml_model == ml_model)
    if energy_source_id is not None:
        query = query.filter(PredictionRecord.energy_source_id == energy_source_id)
    if prediction_source is not None:
        query = query.filter(PredictionRecord.prediction_source == prediction_source)
    if start_date is not None:
        query = query.filter(PredictionRecord.predict_date >= start_date)
    if end_date is not None:
        query = query.filter(PredictionRecord.predict_date <= end_date)
    
    return query.order_by(PredictionRecord.predict_date.desc()).limit(limit).all()
   
        
def get_all_energy_sources(db: Session):
    return db.query(EnergySource).all()

 
def create_database_if_not_exists(db_url):
    db_name = db_url.rsplit("/", 1)[-1] 
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        conn.execute(text("COMMIT"))

        # Check if DB exists
        result = conn.execute(
            text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        ).fetchone()

        if not result:
            conn.execute(text(f"CREATE DATABASE {db_name} TEMPLATE template0"))
            print(f"Database '{db_name}' created.")
        else:
            print(f"Database '{db_name}' already exists.")