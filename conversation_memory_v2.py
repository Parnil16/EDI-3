# conversation_memory_v2.py

import os
import sqlite3
import logging
import json
import random
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from filelock import FileLock, Timeout # <-- Import Timeout

# --- Setup logging ---
logger = logging.getLogger("ConversationMemory")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[ConversationMemory] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ConversationMemory:
    """
    Manages persistent, thread-safe conversation history and feedback using SQLite.
    
    v2 Improvements:
    - (Fix #6) Enabled WAL mode for high read/write concurrency.
    - (Fix #5) Upgraded schema to include confidence and model version.
    - (Fix #4) Hardened JSON serialization against np.float types.
    - (Fix #8) Added get_recent_emotional_vector() for ML integration.
    
    FIX APPLIED (v2.1):
    - Added FileLock to all read methods to prevent 'database is locked' errors (Flaw 5).
    """

    BASE_DIR = "conversation_data"
    DB_SCHEMA_VERSION = 2  # <-- PATCH: Schema bump
    MAX_ROWS = 2000  # Pruning limit
    PRUNE_CHANCE = 0.05 # Run prune check 5% of the time

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        os.makedirs(self.BASE_DIR, exist_ok=True)
        
        self.db_path = os.path.join(self.BASE_DIR, f"{self.user_id}_memory.sqlite")
        self.lock_path = os.path.join(self.BASE_DIR, f"{self.user_id}.lock")

        self._init_db()

    def _init_db(self):
        """Initializes the database, schema, and version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                
                # --- PATCH (Fix #6): Enable WAL Mode ---
                conn.execute("PRAGMA journal_mode=WAL;")
                
                # Check and set schema version
                cursor = conn.execute("PRAGMA user_version")
                current_version = cursor.fetchone()[0]
                
                if 0 < current_version < self.DB_SCHEMA_VERSION:
                    logger.warning(f"DB schema mismatch: Found v{current_version}, expected v{self.DB_SCHEMA_VERSION}.")
                    # Note: A real migration (Fix #1) would go here.
                conn.execute(f"PRAGMA user_version = {self.DB_SCHEMA_VERSION}")

                # --- PATCH (Fix #5): Upgraded Schema ---
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_msg TEXT NOT NULL,
                        bot_resp TEXT NOT NULL,
                        style_label TEXT,
                        feedback_score INTEGER DEFAULT 0 CHECK(feedback_score IN (-1, 0, 1)),
                        emotions_json TEXT,
                        predicted_mood TEXT,
                        confidence_score REAL,
                        model_version TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions (timestamp)")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to initialize DB at {self.db_path}: {e}", exc_info=True)

    def _prune_db(self, conn: sqlite3.Connection):
        """Deletes the oldest records if the database exceeds MAX_ROWS."""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM interactions")
            count = cursor.fetchone()[0]
            
            if count > self.MAX_ROWS:
                rows_to_delete = count - self.MAX_ROWS
                logger.info(f"Pruning {rows_to_delete} old interactions...")
                cursor.execute("""
                    DELETE FROM interactions WHERE id IN (
                        SELECT id FROM interactions ORDER BY timestamp ASC LIMIT ?
                    )
                """, (rows_to_delete,))
                conn.commit()
                logger.info("Pruning complete.")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to prune database: {e}", exc_info=True)

    def store_interaction(self, user_msg: str, bot_resp: str, 
                        style_label: Optional[str], 
                        emotions_dict: Optional[Dict[str, float]], 
                        predicted_mood: Optional[str],
                        confidence_score: Optional[float] = None,  # <-- PATCH (Fix #5)
                        model_version: Optional[str] = None) -> Optional[int]: # <-- PATCH (Fix #5)
        """
        Stores a complete interaction turn in the database (thread-safe).
        """
        ts = datetime.utcnow().isoformat()
        
        # --- PATCH (Fix #4): Use default=float to handle np.float32 etc. ---
        emotions_json = json.dumps(emotions_dict, default=float) if emotions_dict else None
        
        try:
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path, timeout=5) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO interactions 
                        (timestamp, user_msg, bot_resp, style_label, emotions_json, 
                         predicted_mood, confidence_score, model_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ts, user_msg, bot_resp, style_label, emotions_json, 
                          predicted_mood, confidence_score, model_version))
                    conn.commit()
                    row_id = cursor.lastrowid

                    if random.random() < self.PRUNE_CHANCE:
                        self._prune_db(conn)
                        
                    return row_id
        except (sqlite3.OperationalError, TimeoutError, Exception) as e: # <-- Added TimeoutError
            logger.error(f"Failed to store interaction: {e}", exc_info=True)
            return None

    def update_feedback(self, interaction_id: int, feedback_score: int):
        """
        Updates feedback for a specific interaction (e.g., +1 for good, -1 for bad).
        """
        safe_score = max(-1, min(feedback_score, 1))
        
        try:
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path, timeout=5) as conn:
                    conn.execute("""
                        UPDATE interactions SET feedback_score = ? WHERE id = ?
                    """, (safe_score, interaction_id))
                    conn.commit()
                    logger.info(f"Updated feedback for interaction {interaction_id} to {safe_score}")
        except (sqlite3.OperationalError, TimeoutError, Exception) as e: # <-- Added TimeoutError
            logger.error(f"Failed to update feedback for {interaction_id}: {e}", exc_info=True)

    def get_recent_history(self, n: int = 10) -> List[Dict[str, str]]:
        """
        Gets the last N conversation turns in the format required by Persona.py.
        """
        history = []
        try:
            # --- FIX 5: Add FileLock ---
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT user_msg, bot_resp FROM interactions ORDER BY id DESC LIMIT ?
                    """, (n,))
                    rows = cursor.fetchall()
                    
                    for row in reversed(rows):
                        history.append({"role": "user", "content": row["user_msg"]})
                        history.append({"role": "assistant", "content": row["bot_resp"]})
                    return history
        except (sqlite3.OperationalError, TimeoutError) as e: # <-- Added TimeoutError
            logger.error(f"Failed to get recent history: {e}", exc_info=True)
            return []

    def get_preferred_styles(self) -> Dict[str, int]:
        """
        Calculates preferred styles by summing feedback scores.
        """
        try:
            # --- FIX 5: Add FileLock ---
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT style_label, SUM(feedback_score) as total_score
                        FROM interactions
                        WHERE style_label IS NOT NULL AND style_label != ''
                        GROUP BY style_label
                        ORDER BY total_score DESC
                    """)
                    rows = cursor.fetchall()
                    
                    if not rows:
                        return {"empathetic": 1}
                    
                    return {row["style_label"]: row["total_score"] for row in rows}
        except (sqlite3.OperationalError, TimeoutError) as e: # <-- Added TimeoutError
            logger.error(f"Failed to get preferred styles: {e}", exc_info=True)
            return {"empathetic": 1}

    # --- PATCH (Fix #8): New ML Vectorization Bridge ---
    def get_recent_emotional_vector(self, n: int = 5) -> Dict[str, float]:
        """
        Calculates the average emotional vector from the last N interactions.
        This is the bridge to the neuro-fuzzy system.
        """
        totals = defaultdict(float)
        count = 0
        try:
            # --- FIX 5: Add FileLock ---
            with FileLock(self.lock_path, timeout=5):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT emotions_json FROM interactions ORDER BY id DESC LIMIT ?", (n,)
                    )
                    rows = cursor.fetchall()
                    
                    for (json_str,) in rows:
                        if not json_str:
                            continue
                        try:
                            emotions = json.loads(json_str)
                            if isinstance(emotions, dict):
                                count += 1
                                for key, value in emotions.items():
                                    totals[key] += float(value)
                        except (json.JSONDecodeError, TypeError):
                            continue # Skip malformed JSON
            
            if count == 0:
                return {} # No data
                
            # Return the averaged vector
            return {key: total / count for key, total in totals.items()}
            
        except (sqlite3.OperationalError, TimeoutError) as e: # <-- Added TimeoutError
            logger.error(f"Failed to get emotional vector: {e}", exc_info=True)
            return {}
    # --- END OF PATCH ---

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing ConversationMemory v2 (SQLite, WAL) ---")
    
    test_db = os.path.join(ConversationMemory.BASE_DIR, "test_user_v2_memory.sqlite")
    test_lock = os.path.join(ConversationMemory.BASE_DIR, "test_user_v2.lock")
    if os.path.exists(test_db): os.remove(test_db)
    if os.path.exists(test_lock): os.remove(test_lock)

    memory = ConversationMemory("test_user_v2")

    print("Storing interactions with new schema...")
    id1 = memory.store_interaction("Hello", "Hi there!", "friendly", {"joy": 0.8}, "happy", 0.95, "v1.0")
    id2 = memory.store_interaction("I'm sad", "I'm sorry.", "empathetic", {"sadness": 0.9}, "sad", 0.88, "v1.0")
    id3 = memory.store_interaction("Thanks", "No problem.", "friendly", {"joy": 0.6}, "happy", 0.91, "v1.0")

    print("\nRecent emotional vector (avg of last 3):")
    vector = memory.get_recent_emotional_vector(n=3)
    print(vector)
    # (0.8 + 0.6) / 2 = 0.7 --- sadness is only 1/3, but joy is 2/3
    # Correct math: (0.8 + 0 + 0.6) / 3 = 0.466
    # (0 + 0.9 + 0) / 3 = 0.3
    assert abs(vector['joy'] - ((0.8 + 0.6) / 3)) < 0.01
    assert abs(vector['sadness'] - (0.9 / 3)) < 0.01
    
    print("\nPreferred styles:")
    memory.update_feedback(id1, 1)
    memory.update_feedback(id2, -1)
    styles = memory.get_preferred_styles()
    print(styles)
    assert styles["friendly"] == 1
    assert styles["empathetic"] == -1

    # Clean up
    os.remove(test_db)
    os.remove(test_lock)
    print("\nCleanup complete.")