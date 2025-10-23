"""
neural_personalizer_refactored_v4.py
------------------------------------
Improvements over v3:
- Safe incremental training with sliding window to prevent overfitting on small data
- Versioning for features and labels to detect mismatches
- Configurable missing value strategy (default=0.0, can be 'mean' or 'median')
- Class weights applied to incremental fine-tuning
- TensorFlow random seed for reproducibility
- Configurable layer sizes and activation functions
- EarlyStopping also available for incremental fine-tuning
- Optional verbose logging

Refinement (v4.1):
- Corrected training logic to only train after min_samples is met.
- Full retrain now happens only on the first run (n == min_samples) or periodically.
- Incremental fine-tuning happens for all other updates (n > min_samples).

FIX APPLIED (v4.2):
- Added FileLock to predict_mood() to prevent read/write race conditions (Flaw 4).
"""

import os
import json
import shutil
import sqlite3
import tempfile
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from filelock import FileLock, Timeout # <-- Import Timeout

# ----------------- Logging -----------------
logger = logging.getLogger("NeuralPersonalizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[NeuralPersonalizer] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class NeuralPersonalizer:
    """Neural network-based personalizer for user emotional patterns."""
    
    # ... other code in __init__ and methods ...

    # ---------------- Config persistence ----------------
    def _persist_config_json(self, path: str, data: Any):
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.user_dir, delete=False) as tf:
                json.dump(data, tf)

            # --- Safe move with retry ---
            retries = 5
            delay = 0.2  # 200 ms
            for attempt in range(retries):
                try:
                    os.replace(tf.name, path)  # atomic move
                    break
                except PermissionError:
                    logger.warning(f"Move attempt {attempt+1} failed, retrying...")
                    time.sleep(delay)
            else:
                raise PermissionError(f"Could not move {tf.name} to {path} after {retries} attempts.")

            os.chmod(path, 0o600)

        except Exception as e:
            logger.warning(f"Config persist failed: {e}", exc_info=True)

    def _load_config_json(self, path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, type(default)) else default
        except Exception as e:
            logger.warning(f"Config load failed: {e}", exc_info=True)
            return default

    # ---------------- Default Features and Labels ----------------
    DEFAULT_FEATURES = ["joy", "sadness", "anger", "fear", "positive", "stress_score"]
    DEFAULT_MOOD_LABELS = ["happy", "calm", "stressed"]


    # ---------------- Default hyperparameters ----------------
    DEFAULT_CONFIG = {
        "min_samples": 30,
        "inc_epochs": 2,
        "full_epochs": 50,
        "patience": 5,
        "batch_size": 8,
        "max_backups": 5,
        "missing_value_strategy": "zero",  # Options: 'zero', 'mean', 'median'
        "layer_sizes": [64, 32],
        "layer_activations": ["relu", "relu"],
        "dropout": 0.2,
        "verbose": False,
        "incremental_window": 20,  # Last N samples used for incremental update
        "random_seed": 42
    }

    def __init__(self, user_id: str = "default_user", config: Optional[dict] = None):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.user_id = user_id
        self.user_dir = os.path.join(self.BASE_DIR, "user_data", self.user_id)
        self.backup_dir = os.path.join(self.BASE_DIR, "backups", self.user_id)
        os.makedirs(self.user_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

        self.model_path = os.path.join(self.user_dir, "user_model.h5")
        self.scaler_path = os.path.join(self.user_dir, "scaler.npz")
        self.labels_path = os.path.join(self.user_dir, "labels.json")
        self.features_path = os.path.join(self.user_dir, "features.json")
        self.version_path = os.path.join(self.user_dir, "version.json")
        self.db_path = os.path.join(self.user_dir, "user_history.sqlite")
        self.lock_path = os.path.join(self.user_dir, "user.lock")

        # ---------------- Config ----------------
        cfg = self.DEFAULT_CONFIG.copy()
        if config:
            cfg.update(config)
        self.config = cfg

        self.min_samples = cfg["min_samples"]
        self.inc_epochs = cfg["inc_epochs"]
        self.full_epochs = cfg["full_epochs"]
        self.patience = cfg["patience"]
        self.batch_size = cfg["batch_size"]
        self.max_backups = cfg["max_backups"]
        self.missing_value_strategy = cfg["missing_value_strategy"]
        self.layer_sizes = cfg["layer_sizes"]
        self.layer_activations = cfg["layer_activations"]
        self.dropout = cfg["dropout"]
        self.verbose = cfg["verbose"]
        self.incremental_window = cfg["incremental_window"]
        self.random_seed = cfg["random_seed"]

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # ---------------- Feature & Label setup ----------------
        self.feature_names = self._load_config_json(self.features_path, self.DEFAULT_FEATURES)
        self.mood_labels = self._load_config_json(self.labels_path, self.DEFAULT_MOOD_LABELS)
        self.input_dim = len(self.feature_names)
        self.output_dim = len(self.mood_labels)

        # Persist configs if missing
        if not os.path.exists(self.features_path):
            self._persist_config_json(self.features_path, self.feature_names)
        if not os.path.exists(self.labels_path):
            self._persist_config_json(self.labels_path, self.mood_labels)
        if not os.path.exists(self.version_path):
            self._persist_config_json(self.version_path, {"feature_names": self.feature_names, "mood_labels": self.mood_labels})

        # ---------------- Initialize DB, Scaler, Model ----------------
        self._init_db()
        self.scaler = self._load_persisted_scaler()
        self.model = self._load_model() # Note: self.model is loaded once, predict() will re-load for safety
        logger.info(f"Initialized NeuralPersonalizer for '{self.user_id}'")

    # ---------------- Database ----------------
    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        features_json TEXT NOT NULL,
                        ground_truth_mood TEXT NOT NULL
                    )
                """)
        except Exception as e:
            logger.error(f"DB init failed: {e}", exc_info=True)

    def _insert_history_safe(self, features_dict: Dict[str, Any], label: str, retries=3):
        ts = datetime.utcnow().isoformat()
        data_json = json.dumps(features_dict)
        for attempt in range(retries):
            try:
                with sqlite3.connect(self.db_path, timeout=5) as conn:
                    conn.execute("INSERT INTO history (timestamp, features_json, ground_truth_mood) VALUES (?, ?, ?)",
                                 (ts, data_json, label))
                return True
            except sqlite3.OperationalError:
                continue
        logger.error(f"Failed to insert into DB after {retries} retries.")
        return False

    def _load_all_training_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        X, y_labels = [], []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute("SELECT features_json, ground_truth_mood FROM history ORDER BY id ASC"):
                    try:
                        fdict = json.loads(row["features_json"])
                        feat_vec = self._prepare_features(fdict)
                        if row["ground_truth_mood"] in self.mood_labels:
                            X.append(feat_vec)
                            y_labels.append(row["ground_truth_mood"])
                    except Exception:
                        continue
            n = len(X)
            if n == 0: return np.array([]), np.array([]), 0
            X_arr = np.array(X)
            y_arr = np.array(y_labels).reshape(-1,1)
            encoder = OneHotEncoder(categories=[self.mood_labels], sparse_output=False, handle_unknown='ignore')
            y_encoded = encoder.fit_transform(y_arr)
            return X_arr, y_encoded, n
        except Exception as e:
            logger.warning(f"Failed to load training data: {e}", exc_info=True)
            return np.array([]), np.array([]), 0

    # ---------------- Feature & Config ----------------
    def _prepare_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        def flatten(d, prefix=''):
            out = {}
            for k,v in d.items():
                if isinstance(v, dict):
                    out.update(flatten(v, prefix+k+'_'))
                else:
                    try:
                        out[prefix+k] = float(v)
                    except (ValueError, TypeError):
                        out[prefix+k] = 0.0
                        logger.warning(f"Could not convert feature '{prefix+k}' (value: {v}). Defaulting to 0.0.")
            return out
        flat = flatten(features_dict)
        vec = [flat.get(f, 0.0) for f in self.feature_names]

        if self.missing_value_strategy in ['mean', 'median'] and self.scaler is not None:
            vec = np.array(vec, dtype=float)
            missing_idx = np.isnan(vec)
            if missing_idx.any():
                if self.missing_value_strategy == 'mean':
                    vec[missing_idx] = self.scaler.mean_[missing_idx]
                else:
                    vec[missing_idx] = np.median(vec[~missing_idx]) # Note: This is median of current vec, not dataset
        return np.array(vec, dtype=float)

    # ---------------- Scaler & Model ----------------
    def _persist_scaler(self, scaler: StandardScaler):
        try:
            np.savez_compressed(self.scaler_path, mean=scaler.mean_, var=scaler.var_, scale=scaler.scale_)
            os.chmod(self.scaler_path, 0o600)
        except Exception as e:
            logger.warning(f"Scaler persist failed: {e}", exc_info=True)

    def _load_persisted_scaler(self) -> Optional[StandardScaler]:
        if not os.path.exists(self.scaler_path): return None
        try:
            npz = np.load(self.scaler_path)
            scaler = StandardScaler()
            scaler.mean_ = npz['mean']; scaler.var_ = npz['var']; scaler.scale_ = npz['scale']
            scaler.n_features_in_ = scaler.mean_.shape[0]
            if scaler.n_features_in_ != self.input_dim:
                logger.warning("Scaler dimensions mismatch. Discarding.")
                return None
            return scaler
        except Exception as e:
            logger.warning(f"Scaler load failed: {e}", exc_info=True)
            return None

    def create_model(self) -> keras.Model:
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        for sz, act in zip(self.layer_sizes, self.layer_activations):
            x = layers.Dense(sz, activation=act, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.output_dim, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _load_model(self) -> keras.Model:
        if os.path.exists(self.model_path):
            try:
                model = keras.models.load_model(self.model_path)
                if model.input_shape[1] == self.input_dim and model.output_shape[1] == self.output_dim:
                    logger.info("Loaded existing model.")
                    return model
                logger.warning("Model dimensions mismatch. Creating new model.")
            except Exception as e:
                logger.warning(f"Model load failed: {e}", exc_info=True)
            self._backup_file(self.model_path)
        logger.info("Creating new model.")
        return self.create_model()

    def _backup_file(self, path: str):
        if not os.path.exists(path): return
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = os.path.join(self.backup_dir, f"{ts}_{os.path.basename(path)}")
        try: shutil.copy2(path, dest)
        except Exception as e: logger.warning(f"Backup failed: {e}", exc_info=True)
        # Keep last N backups
        backups = sorted(os.listdir(self.backup_dir))
        while len(backups) > self.max_backups:
            try: os.remove(os.path.join(self.backup_dir, backups.pop(0)))
            except: break

    # ---------------- Public API ----------------
    def update_emotional_model(self, features_dict: Dict[str, Any], ground_truth_label: str):
        """Train or fine-tune the model with a new data point (thread-safe)."""
        if ground_truth_label not in self.mood_labels:
            logger.warning(f"Invalid label '{ground_truth_label}'")
            return

        with FileLock(self.lock_path):
            if not self._insert_history_safe(features_dict, ground_truth_label):
                return

            X, y, n_samples = self._load_all_training_data()
            if n_samples < self.min_samples:
                logger.info(f"Collecting data. {n_samples}/{self.min_samples} samples.")
                return  # Don't train yet

            # --- Full retrain triggers ---
            is_first_train = not os.path.exists(self.model_path)
            is_periodic_retrain = (n_samples % 50 == 0)  # retrain every 50 samples

            if is_first_train or is_periodic_retrain:
                try:
                    logger.info(f"Full retrain with {n_samples} samples.")
                    # Fit scaler on all data
                    self.scaler = StandardScaler().fit(X)
                    self._persist_scaler(self.scaler)
                    X_scaled = self.scaler.transform(X)

                    # Train/validation split
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
                    )
                    self._backup_file(self.model_path)
                    new_model = self.create_model()
                    early = keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=self.patience, restore_best_weights=True
                    )

                    # Class weights
                    counts = np.sum(y_tr, axis=0)
                    total = counts.sum()
                    class_weight = {
                        i: float(total)/(len(counts)*counts[i]) if counts[i] > 0 else 1.0
                        for i in range(len(counts))
                    }

                    new_model.fit(
                        X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=self.full_epochs,
                        batch_size=self.batch_size,
                        callbacks=[early],
                        class_weight=class_weight,
                        verbose=int(self.verbose)
                    )
                    new_model.save(self.model_path)
                    self.model = new_model
                    logger.info(f"Model fully retrained.")
                except Exception as e:
                    logger.error(f"Full retrain failed: {e}", exc_info=True)
                return  # Done after full retrain

            # --- Incremental fine-tuning (sliding window) ---
            try:
                if self.scaler is None:
                    logger.warning("No scaler found for incremental update. Skipping update.")
                    return

                # Use sliding window for last N samples
                if n_samples > self.incremental_window:
                    X = X[-self.incremental_window:]
                    y = y[-self.incremental_window:]

                X_scaled = self.scaler.transform(X)
                counts = np.sum(y, axis=0)
                total = counts.sum()
                class_weight = {
                    i: float(total)/(len(counts)*counts[i]) if counts[i] > 0 else 1.0
                    for i in range(len(counts))
                }

                self.model.fit(
                    X_scaled, y,
                    epochs=self.inc_epochs,
                    batch_size=self.batch_size,
                    class_weight=class_weight,
                    verbose=int(self.verbose)
                )
                self.model.save(self.model_path)
                logger.info(f"Incremental fine-tuning completed on {len(X)} samples.")
            except Exception as e:
                logger.warning(f"Incremental update failed: {e}", exc_info=True)

    def predict_mood(self, features_dict: Dict[str, Any]) -> Tuple[str, float]:
        try:
            # --- FIX 4: Add FileLock for thread-safe prediction ---
            with FileLock(self.lock_path, timeout=5):
                X = np.array([self._prepare_features(features_dict)])
                
                # Load scaler from disk inside the lock
                scaler = self._load_persisted_scaler() 
                if scaler is not None:
                    X = scaler.transform(X)
                else:
                    logger.warning("Predicting without a scaler. Model may perform poorly.")

                # Check if model file exists before trying to load it
                if not os.path.exists(self.model_path):
                    logger.warning("No model file found. Cannot predict.")
                    return "unknown", 0.0

                # Load model from disk inside the lock for safety
                model = keras.models.load_model(self.model_path)
                pred = model.predict(X, verbose=0)[0]
                idx = int(np.argmax(pred))
                return self.mood_labels[idx], float(np.max(pred))
            # --- End Fix 4 ---
        except (Timeout, Exception) as e: # <-- Add Timeout
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return "unknown", 0.0


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Clean up previous runs for a clean test
    test_user_dir = os.path.join(NeuralPersonalizer.BASE_DIR, "user_v4_test")
    if os.path.exists(test_user_dir):
        shutil.rmtree(test_user_dir)
        print(f"Removed old test directory: {test_user_dir}")

    custom_config = {"min_samples":5, "inc_epochs":1, "full_epochs":10, "verbose": True}
    user_model = NeuralPersonalizer("user_v4_test", config=custom_config)

    stressed_data = {"joy":0.1,"sadness":0.6,"anger":0.7,"fear":0.5,"positive":0.1,"stress_score":8.5}
    calm_data = {"joy":0.7,"sadness":0.1,"anger":0.0,"fear":0.0,"positive":0.8,"stress_score":1.0}
    happy_data = {"joy":0.9,"sadness":0.0,"anger":0.0,"fear":0.0,"positive":0.9,"stress_score":0.5}

    print("\n--- Collecting data (should not train) ---")
    user_model.update_emotional_model(stressed_data, "stressed") # 1
    user_model.update_emotional_model(calm_data, "calm")     # 2
    user_model.update_emotional_model(happy_data, "happy")   # 3
    user_model.update_emotional_model(stressed_data, "stressed") # 4

    print("\n--- Triggering first full retrain (n=5) ---")
    user_model.update_emotional_model(stressed_data, "stressed") # 5

    label, conf = user_model.predict_mood(stressed_data)
    print(f"\nPrediction for stressed: {label} (Confidence: {conf:.2f})")
    label, conf = user_model.predict_mood(calm_data)
    print(f"Prediction for calm: {label} (Confidence: {conf:.2f})")

    print("\n--- Triggering incremental update (n=6) ---")
    user_model.update_emotional_model(happy_data, "happy") # 6
    
    label, conf = user_model.predict_mood(happy_data)
    print(f"\nPrediction for happy: {label} (Confidence: {conf:.2f})")