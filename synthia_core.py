# synthia_core.py

import logging
import threading  # <-- 1. IMPORT THREADING
import re
from typing import Dict, Tuple
from datetime import datetime

# --- 2. FIX: USE CORRECT RELATIVE IMPORTS ---
try:
    from .neural_personalizer import NeuralPersonalizer
    from .conversation_memory_v2 import ConversationMemory
    from .behavior_tracker import BehaviorTracker
    from Persona import build_prompt
    from Fuzzy_delay import apply_delay
except ImportError:
    # Fallback to absolute imports for script mode
    # This allows running the file directly for testing
    logging.warning("Relative imports failed. Falling back to script-mode imports.")
    from neural_personalizer import NeuralPersonalizer
    from conversation_memory_v2 import ConversationMemory
    from behavior_tracker import BehaviorTracker
    from Persona import build_prompt
    from Fuzzy_delay import apply_delay
# ------------------------------------------------------------------

# Note: Keras/TF import is removed as K.clear_session() is no longer used

# --- Logging Setup ---
logger = logging.getLogger("SynthiaCore") 
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[SynthiaCore] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SynthiaCore:
    """
    Integrates neural personalization, conversation memory, and behavior tracking
    to generate intelligent and adaptive responses.
    """

    def __init__(self, user_id: str = "default_user", llm=None, 
                 sentiment_analyzer=None, emotion_analyzer=None, llm_lock=None): # <-- 3. ACCEPT LLM_LOCK
        self.user_id = user_id
        
        # LLMs and Pipelines
        self.llm = llm
        self.llm_lock = llm_lock  # <-- 4. STORE GLOBAL LLM_LOCK
        self.sentiment_analyzer = sentiment_analyzer
        self.emotion_analyzer = emotion_analyzer

        # Initialize modules
        self.personalizer = NeuralPersonalizer(user_id)
        self.conv_memory = ConversationMemory(user_id)
        self.behavior_tracker = BehaviorTracker(user_id)
        
        # Chat history
        self.chat_history = []
        
        # State attributes
        self.last_interaction_time = datetime.utcnow()
        self.last_user_features = None 
        self.last_feedback_time = datetime.min
        
        # --- 5. INITIALIZE PER-USER LOCKS ---
        # Protects all instance attributes (chat_history, last_features) during a single request
        self.generation_lock = threading.Lock()
        # Protects last_user_features from being read by feedback while being written by generate_response
        self.feature_lock = threading.Lock()
        # ----------------------------------------

        logger.info(f"SynthiaCore initialized for user '{self.user_id}'")

    # ---------------- Helper Methods ----------------
    
    def _analyze_text_with_scores(self, text):
        """Analyzes text for sentiment and emotion scores with safe fallbacks."""
        # (No changes to this method)
        try:
            sentiment_result = self.sentiment_analyzer(text)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            sentiment_result = [{"label": "neutral", "score": 0.5}]
        try:
            emotion_result = self.emotion_analyzer(text, top_k=None)
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            emotion_result = [{"label": "neutral", "score": 0.5}]

        sentiment_scores = {item['label'].lower(): item['score'] for item in sentiment_result}
        positive_score = sentiment_scores.get("positive", 1 - sentiment_scores.get("negative", 0.5))
        sentiment_dict = {"positive": positive_score}
        emotion_dict = {item['label'].lower(): item['score'] for item in emotion_result}
        return sentiment_dict, emotion_dict

    def _determine_response_style(self, scores):
        """Sorts emotions to find the top 3."""
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {name: score for name, score in sorted_emotions[:3]}

    # ---------------- Core Response Generation ----------------

    def generate_response(self, user_message: str) -> Tuple[str, dict, str, float]:
        """
        Generates a full response by processing emotion, applying delay,
        and querying the LLM.
        """
        # --- 6. USE PER-USER LOCK FOR ATOMICITY ---
        with self.generation_lock:
            # 1. Analyze sentiment & emotion
            sentiment_scores, emotion_scores = self._analyze_text_with_scores(user_message)
            top_emotions = self._determine_response_style(emotion_scores)

            # 3. ACTIVATE NEURAL PERSONALIZER (PREDICTION)
            
            # --- 7. INTEGRATE BEHAVIOR TRACKER DATA ---
            long_term_behavior = self.behavior_tracker.get_behavioral_summary_vector()
            
            raw_stress = emotion_scores.get("anger", 0.0) * 10
            
            # Combine instant, long-term, and raw features
            features_for_nn = {
                "joy": emotion_scores.get("joy", 0.0),
                "sadness": emotion_scores.get("sadness", 0.0),
                "anger": emotion_scores.get("anger", 0.0),
                "fear": emotion_scores.get("fear", 0.0),
                "positive": sentiment_scores.get("positive", 0.5),
                "stress_score": min(max(raw_stress, 0.0), 10.0),
                
                # Add normalized long-term features
                "avg_stress": long_term_behavior.get("norm_stress", 0.0),
                "avg_msg_len": long_term_behavior.get("norm_msg_length", 0.0),
                "avg_time_gap": long_term_behavior.get("norm_time_gap", 0.0),
                "avg_engagement": long_term_behavior.get("norm_engagement", 0.0)
            }
            
            with self.feature_lock:
                self.last_user_features = features_for_nn
            # --- END OF FEATURE LOCK ---

            # 3. Predict mood using the NN
            predicted_mood, confidence = self.personalizer.predict_mood(features_for_nn)
            logger.info(f"Predicted mood: {predicted_mood} (Conf: {confidence:.2f})")
            # --- DEFAULT MODE FALLBACK ---
            if predicted_mood == "unknown" or confidence < 0.3:
                predicted_mood = "neutral"  # or "calm" if you prefer
                confidence = 0.5
                logger.info(f"Using default mood: {predicted_mood} (Conf: {confidence:.2f})")


            # 4. Use prediction to inform fuzzy delay
            if predicted_mood == "stressed":
                stress_score_for_delay = max(features_for_nn["stress_score"], 8.0) 
            else:
                stress_score_for_delay = features_for_nn["stress_score"]

            # 5. Check for importance keywords
            important_keywords = ["help", "urgent", "emergency", "important"]
            important = any(word in user_message.lower() for word in important_keywords)

            # 6. Apply fuzzy delay
            apply_delay(
                user_message,
                stress_score_for_delay,
                important=important,
                debug=True,
                show_debug=True,
                show_progress=True
            )

            # 7. Build LLM prompt
            prompt = build_prompt(
                chat_history=self.chat_history[-10:],
                user_message=user_message
            )
            
            # 8. Update chat history
            self.chat_history.append({"role": "user", "content": user_message})

            # 9. Generate LLM reply
            try:
                llm_call = None
                if hasattr(self.llm, "invoke"):
                    llm_call = lambda: self.llm.invoke(prompt, stop=["You:", "User:", "Synthia:", "\n\n"]).strip()
                elif callable(self.llm):
                    llm_call = lambda: self.llm(prompt).strip()
                else:
                    raise AttributeError("LLM object has no valid invoke method.")

                # --- 8. USE GLOBAL LLM_LOCK ---
                if self.llm_lock:
                    with self.llm_lock:
                        llm_reply = llm_call()
                else:
                    # Fallback if no lock is provided
                    llm_reply = llm_call()
                # --- END GLOBAL LOCK ---

            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)
                llm_reply = "I'm having a momentary lapse in thought — could you repeat that?"

            # 10. Clean up reply and strip unwanted narrative or meta text
            import re

            # Remove everything after narrative or system markers
            narrative_markers = [
                r"\.\.\.and the conversation continues",
                r"Synthia listens attentively",
                r"Synthia is always listening",
                r"ultimately striving to create",
                r"providing a nurturing environment",
            ]
            for marker in narrative_markers:
                llm_reply = re.split(marker, llm_reply, flags=re.IGNORECASE)[0]

            # Remove redundant speaker labels or meta prefixes
            if "Synthia:" in llm_reply:
                llm_reply = llm_reply.split("Synthia:", 1)[-1]
            if "User:" in llm_reply:
                llm_reply = llm_reply.split("User:", 1)[0]

            # Final cleanup
            llm_reply = llm_reply.strip()


            # 11. Add LLM reply to history
            self.chat_history.append({"role": "assistant", "content": llm_reply})

            # Auto-prune chat history
            if len(self.chat_history) > 100:
                self.chat_history = self.chat_history[-50:]

            # 12. Store conversation & behavior
            preferred_styles = self.conv_memory.get_preferred_styles()
            style_key = list(preferred_styles.keys())[0] if preferred_styles else "default"
            
            # --- 9. FIX: Call store_interaction with all correct parameters ---
            self.conv_memory.store_interaction(
                user_msg=user_message,
                bot_resp=llm_reply,
                style_label=style_key,
                emotions_dict=emotion_scores,
                predicted_mood=predicted_mood,
                confidence_score=confidence,
                model_version="v_core_1.0"
            )
            # --- END OF FIX ---
            
            current_time = datetime.utcnow()
            time_since_last = (current_time - self.last_interaction_time).total_seconds()
            self.last_interaction_time = current_time

            self.behavior_tracker.record_behavior(
                stress_level=features_for_nn["stress_score"],
                message_length=len(user_message),
                time_since_last_message=time_since_last,
                engagement_score=1.0
            )

            return llm_reply, top_emotions, predicted_mood, round(confidence, 2)
        # --- END OF GENERATION LOCK ---


    def record_mood_feedback(self, ground_truth_label: str):
        """
        Records user feedback to train the neural personalizer.
        This should be called when the user confirms their mood.
        """
        if (datetime.utcnow() - self.last_feedback_time).total_seconds() < 5:
            logger.info("Skipping feedback retrain (cooldown active).")
            return
        self.last_feedback_time = datetime.utcnow()

        if ground_truth_label not in self.personalizer.mood_labels:
            logger.warning(f"Invalid mood label '{ground_truth_label}' for training.")
            return

        features_to_train = None
        with self.feature_lock:
            if self.last_user_features is not None:
                features_to_train = self.last_user_features
                self.last_user_features = None
        
        if features_to_train is None:
            logger.warning("No recent user features to associate with feedback.")
            return

        try:
            self.personalizer.update_emotional_model(
                features_to_train, 
                ground_truth_label
            )
            # --- 10. REMOVED K.clear_session() ---
            logger.info(f"Recorded feedback: '{ground_truth_label}' for last features.")
        except Exception as e:
            logger.error(f"Failed to update emotional model: {e}", exc_info=True)
    # ---------------------------------------------------


    # ---------------- Module Passthrough Methods ----------------
    
    def record_conversation_feedback(self, interaction_id: int, feedback: int):
        """Updates feedback for a specific interaction."""
        self.conv_memory.update_feedback(interaction_id, feedback)
        logger.info(f"Passed feedback {feedback} to memory for id {interaction_id}")