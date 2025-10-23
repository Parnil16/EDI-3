# behavior_tracker_v4.py

import os
import sqlite3
import logging
import random
from datetime import datetime
from typing import Dict, Optional, Tuple
from filelock import FileLock, Timeout # <-- Import Timeout

# --- Setup logging ---
logger = logging.getLogger("BehaviorTracker")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[BehaviorTracker] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class BehaviorTracker:
    """
    Tracks user behavior patterns such as stress, message timing, and engagement.

    Version 4 Improvements:
    - Added Exponential Moving Average (EMA) for stable, smoothed output vectors.
    - Added schema version check on init to warn of migrations.
    - (v3) Input clamping and normalization for ML-readiness.
    - (v3) Automatic database pruning to prevent infinite growth.
    - (v3) Schema versioning and CHECK constraints for data integrity.
    
    FIX APPLIED (v4.1):
    - Added FileLock to all read methods to prevent 'database is locked' errors (Flaw 5).
    """

    BASE_DIR = "behavior_data"
    DB_SCHEMA_VERSION = 1
    
    # Configurable limits
    MAX_ROWS = 5000
    PRUNE_CHANCE = 0.1 # Only run prune check 10% of the time

    # Limits for clamping inputs
    CLAMP_LIMITS = {
        "stress_level": (0.0, 10.0),
        "message_length": (0, 5000),
        "time_since_last_message": (0.0, 3600.0), # 1 hour max gap
        "engagement_score": (0.0, 1.0)
    }

    # Limits for normalizing the summary vector
    NORM_LIMITS = {
        "stress": 10.0,
        "msg_length": 250.0, # Avg msg length > 250 is max
        "time_gap": 120.0,   # Avg time gap > 2 mins is max
        "engagement": 1.0
    }

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        os.makedirs(self.BASE_DIR, exist_ok=True)

        self.db_path = os.path.join(self.BASE_DIR, f"{self.user_id}_behavior.sqlite")
        self.lock_path = os.path.join(self.BASE_DIR, f"{self.user_id}.lock")
        
        # --- PATCH: Add EMA vector cache ---
        self.prev_vector = {}
        # -----------------------------------

        self._init_db()

    def _init_db(self):
        """Initializes the behavior database, schema, and version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                
                # --- PATCH: Schema Version Check ---
                cursor = conn.execute("PRAGMA user_version")
                current_version = cursor.fetchone()[0]
                
                if 0 < current_version < self.DB_SCHEMA_VERSION:
                    logger.warning(f"DB schema mismatch: Found v{current_version}, expected v{self.DB_SCHEMA_VERSION}. "
                                 f"Consider running a migration.")
                elif current_version > self.DB_SCHEMA_VERSION:
                     logger.error(f"DB schema downgrade detected: Found v{current_version}, expected v{self.DB_SCHEMA_VERSION}.")
                
                # Set current version
                conn.execute(f"PRAGMA user_version = {self.DB_SCHEMA_VERSION}")
                # --- END OF PATCH ---
                
                # Use CHECK constraints for data integrity (Issue 1)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS behavior (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        stress_level REAL CHECK(stress_level >= 0 AND stress_level <= 10),
                        message_length INTEGER CHECK(message_length >= 0),
                        time_since_last_message REAL CHECK(time_since_last_message >= 0),
                        engagement_score REAL CHECK(engagement_score >= 0 AND engagement_score <= 1)
                    )
                """)
                # Add index for faster pruning and time-series queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON behavior (timestamp)")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to initialize DB at {self.db_path}: {e}", exc_info=True)

    def _clamp_value(self, value: float, key: str) -> float:
        """Clamps a value based on the defined limits."""
        min_val, max_val = self.CLAMP_LIMITS.get(key, (-float('inf'), float('inf')))
        return max(min_val, min(value, max_val))

    def record_behavior(self, stress_level: float, message_length: int,
                        time_since_last_message: float, engagement_score: float,
                        retries: int = 3) -> Optional[int]:
        """
        Inserts a new behavior record, clamping inputs for safety (Issue 2).
        """
        ts = datetime.utcnow().isoformat()
        
        # --- Clamp all inputs before insertion ---
        safe_stress = self._clamp_value(stress_level, "stress_level")
        safe_length = int(self._clamp_value(message_length, "message_length"))
        safe_time_gap = self._clamp_value(time_since_last_message, "time_since_last_message")
        safe_engagement = self._clamp_value(engagement_score, "engagement_score")
        
        for attempt in range(retries):
            try:
                with FileLock(self.lock_path, timeout=5):
                    with sqlite3.connect(self.db_path, timeout=5) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO behavior (timestamp, stress_level, message_length,
                                                  time_since_last_message, engagement_score)
                            VALUES (?, ?, ?, ?, ?)
                        """, (ts, safe_stress, safe_length, safe_time_gap, safe_engagement))
                        conn.commit()
                        row_id = cursor.lastrowid
                        
                        # Trigger pruning check (Issue 1)
                        if random.random() < self.PRUNE_CHANCE:
                            self._prune_db(conn)
                            
                        return row_id
            except (sqlite3.OperationalError, TimeoutError) as e: # <-- Added TimeoutError
                logger.warning(f"Attempt {attempt + 1}/{retries} failed to insert behavior: {e}")
                if attempt + 1 == retries:
                    logger.error("All attempts to insert behavior failed.", exc_info=True)
                    return None
        return None

    def _prune_db(self, conn: sqlite3.Connection):
        """
        Deletes the oldest records if the database exceeds MAX_ROWS (Issue 1).
        This is called by record_behavior, so it's already inside a lock.
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM behavior")
            count = cursor.fetchone()[0]
            
            if count > self.MAX_ROWS:
                rows_to_delete = count - self.MAX_ROWS
                logger.info(f"Pruning {rows_to_delete} old behavior records...")
                # Delete the oldest N records
                cursor.execute("""
                    DELETE FROM behavior WHERE id IN (
                        SELECT id FROM behavior ORDER BY timestamp ASC LIMIT ?
                    )
                """, (rows_to_delete,))
                conn.commit()
                logger.info("Pruning complete.")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to prune database: {e}", exc_info=True)

    def get_behavioral_summary_vector(self) -> Dict[str, float]:
        """
        Computes a *smoothed* and *normalized* feature vector for the ML model.
        Returns a dictionary with all values scaled between 0.0 and 1.0.
        """
        try:
            # --- FIX 5: Add FileLock ---
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT
                            AVG(COALESCE(stress_level, 0.0)),
                            AVG(COALESCE(message_length, 0.0)),
                            AVG(COALESCE(time_since_last_message, 0.0)),
                            AVG(COALESCE(engagement_score, 0.0))
                        FROM behavior
                    """)
                    stats = cursor.fetchone()
            # --- End Lock ---

            raw_vector = {}
            if stats is None or stats[0] is None:
                # Return a default "calm" vector
                raw_vector = {"norm_stress": 0.0, "norm_msg_length": 0.0,
                              "norm_time_gap": 0.0, "norm_engagement": 0.0}
            else:
                # Unpack raw averages
                avg_stress, avg_msg_len, avg_time_gap, avg_engage = stats
                
                # Normalize all values to be [0.0, 1.0] for the ML model
                raw_vector = {
                    "norm_stress": min(avg_stress / self.NORM_LIMITS["stress"], 1.0),
                    "norm_msg_length": min(avg_msg_len / self.NORM_LIMITS["msg_length"], 1.0),
                    "norm_time_gap": min(avg_time_gap / self.NORM_LIMITS["time_gap"], 1.0),
                    "norm_engagement": min(avg_engage / self.NORM_LIMITS["engagement"], 1.0)
                }
            
            # --- PATCH: Warm Start Smoother (EMA) ---
            smoothed_vector = {}
            for k, v in raw_vector.items():
                # Apply smoothing: 80% previous value, 20% new value
                smoothed_vector[k] = (0.8 * self.prev_vector.get(k, v)) + (0.2 * v)
            
            self.prev_vector = smoothed_vector # Cache for next run
            return smoothed_vector
            # --- END OF PATCH ---

        except (sqlite3.OperationalError, TimeoutError) as e: # <-- Added TimeoutError
            logger.error(f"Failed to summarize behavior: {e}", exc_info=True)
            # On error, return the last known good vector or a default
            return self.prev_vector if self.prev_vector else {
                "norm_stress": 0.0, "norm_msg_length": 0.0,
                "norm_time_gap": 0.0, "norm_engagement": 0.0
            }

    def get_recent_behaviors(self, n: int = 20) -> list[Dict]:
        """
        Retrieves the last `n` behavior records in chronological order.
        """
        try:
            # --- FIX 5: Add FileLock ---
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    # Order by ID DESC for index, then reverse in Python for chronology
                    cursor = conn.execute("SELECT * FROM behavior ORDER BY id DESC LIMIT ?", (n,))
                    rows = cursor.fetchall()
            # --- End Lock ---
            
            # Reverse to get chronological order (oldest first)
            return [dict(row) for row in reversed(rows)]
        except (sqlite3.OperationalError, TimeoutError) as e: # <-- Added TimeoutError
            logger.error(f"Failed to fetch recent behaviors: {e}", exc_info=True)
            return []


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing BehaviorTracker v4 ---")
    
    # Clean up old test files if they exist
    test_db_path = os.path.join(BehaviorTracker.BASE_DIR, "test_user_v4_behavior.sqlite")
    test_lock_path = os.path.join(BehaviorTracker.BASE_DIR, "test_user_v4.lock")
    if os.path.exists(test_db_path): os.remove(test_db_path)
    if os.path.exists(test_lock_path): os.remove(test_lock_path)
    
    tracker = BehaviorTracker("test_user_v4")

    # Record some sample behaviors
    print("\nRecording sample behaviors...")
    tracker.record_behavior(7.5, 120, 30.0, 0.8)
    tracker.record_behavior(3.2, 45, 5.0, 0.5)

    # Summarize
    print("\nBehavior summary vector (normalized + smoothed):")
    summary_vec = tracker.get_behavioral_summary_vector()
    for k, v in summary_vec.items():
        print(f"  - {k}: {v:.3f}")
        
    print("\nRecording new behavior...")
    tracker.record_behavior(10.0, 200, 10.0, 1.0)
    
    print("\nNew summary vector (should be smoothed):")
    summary_vec_2 = tracker.get_behavioral_summary_vector()
    for k, v in summary_vec_2.items():
        print(f"  - {k}: {v:.3f}")

    # Clean up test DB
    os.remove(tracker.db_path)
    os.remove(tracker.lock_path)
    print("\nCleanup complete.")