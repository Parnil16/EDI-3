#Fuzzy_delay.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
import time
import sys

# --- Configurable constants ---
IMPORTANT_RANGE = (0.0, 0.8)   # fast response for important messages
MIN_DELAY = 0.2                # minimum delay clamp
JITTER_FACTOR = 0.1            # ±10% randomness

# --- Toggle fuzzy debug/progress features ---
ENABLE_FUZZY_DEBUG = True  # Set False to disable all debug/progress outputs

# --- Define fuzzy inputs and outputs ---
message_length = ctrl.Antecedent(np.arange(0, 301, 1), 'message_length')
stress_level = ctrl.Antecedent(np.arange(0, 11, 1), 'stress_level')
response_delay = ctrl.Consequent(np.arange(0, 6, 0.1), 'response_delay')

# --- Membership functions ---
message_length['short'] = fuzz.trimf(message_length.universe, [0, 0, 100])
message_length['medium'] = fuzz.trimf(message_length.universe, [50, 150, 250])
message_length['long'] = fuzz.trimf(message_length.universe, [150, 300, 300])

stress_level['low'] = fuzz.trimf(stress_level.universe, [0, 0, 4])
stress_level['medium'] = fuzz.trimf(stress_level.universe, [2, 5, 8])
stress_level['high'] = fuzz.trimf(stress_level.universe, [6, 10, 10])

response_delay['immediate'] = fuzz.trimf(response_delay.universe, [0, 0, 1.5])
response_delay['normal'] = fuzz.trimf(response_delay.universe, [1, 2.5, 4])
response_delay['long'] = fuzz.trimf(response_delay.universe, [3, 5, 6])

# --- Rules ---
rule1 = ctrl.Rule(message_length['short'] & stress_level['low'], response_delay['normal'])
rule2 = ctrl.Rule(message_length['short'] & stress_level['medium'], response_delay['normal'])
rule3 = ctrl.Rule(message_length['short'] & stress_level['high'], response_delay['immediate'])
rule4 = ctrl.Rule(message_length['medium'] & stress_level['low'], response_delay['normal'])
rule5 = ctrl.Rule(message_length['medium'] & stress_level['medium'], response_delay['normal'])
rule6 = ctrl.Rule(message_length['medium'] & stress_level['high'], response_delay['immediate'])
rule7 = ctrl.Rule(message_length['long'] & stress_level['low'], response_delay['long'])
rule8 = ctrl.Rule(message_length['long'] & stress_level['medium'], response_delay['normal'])
rule9 = ctrl.Rule(message_length['long'] & stress_level['high'], response_delay['normal'])

# --- Control system ---
delay_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# --- Compute fuzzy delay ---
def compute_delay(msg_text: str, stress_score: float, important: bool = False, debug: bool = False) -> float:
    if important:
        delay_seconds = round(random.uniform(*IMPORTANT_RANGE), 2)
        return (delay_seconds, {"mode": "important"}) if debug else delay_seconds

    sim = ctrl.ControlSystemSimulation(delay_ctrl)
    
    # --- PATCH APPLIED ---
    # Clamp inputs to the defined ranges [0, 300] and [0, 10]
    msg_len = max(0, min(len(msg_text), 300))
    stress = max(0.0, min(stress_score, 10.0))
    # --- END OF PATCH ---

    sim.input['message_length'] = msg_len
    sim.input['stress_level'] = stress

    try:
        sim.compute()
        delay_seconds = sim.output['response_delay']
    except Exception:
        delay_seconds = 1.5  # safe fallback

    # Add jitter
    jitter = random.uniform(-JITTER_FACTOR * delay_seconds, JITTER_FACTOR * delay_seconds)
    delay_seconds = max(delay_seconds + jitter, MIN_DELAY)

    delay_seconds = round(delay_seconds, 2)

    if debug:
        return delay_seconds, {
            "msg_len": msg_len,
            "stress": stress,
            "raw_delay": round(sim.output['response_delay'], 2),
            "jitter": round(jitter, 3),
            "final_delay": delay_seconds,
            "mode": "fuzzy"
        }

    return delay_seconds

# --- Visual debug ---
def fuzzy_visual_debug(msg_text, stress_score, important, debug_info, steps=20):
    if not ENABLE_FUZZY_DEBUG:
        return  # Skip entirely if disabled

    delay_seconds = debug_info['final_delay']
    if delay_seconds <= 0:
        print(f"[Fuzzy Delay] Skipped (0s) - important={important}")
        return

    summary = (
        f"[Fuzzy Delay] {delay_seconds}s | "
        f"Stress={stress_score:.1f} | "
        f"MsgLen={len(msg_text)} | "
        f"Important={important}"
    )
    print(summary)

    step_time = delay_seconds / steps
    sys.stdout.write("[")
    sys.stdout.flush()
    for _ in range(steps):
        time.sleep(step_time)
        sys.stdout.write("=")
        sys.stdout.flush()
    sys.stdout.write("]\n")

# --- Apply delay ---
def apply_delay(msg_text: str, stress_score: float, important: bool = False,
                debug: bool = False, show_debug: bool = False, show_progress: bool = False):
    result = compute_delay(msg_text, stress_score, important, debug)
    delay = result[0] if debug else result

    if ENABLE_FUZZY_DEBUG:
        if show_debug and debug:
            info = result[1]
            print(f"[DEBUG] Message length={info['msg_len']}, Stress={info['stress']}, "
                  f"Raw delay={info['raw_delay']}, Jitter={info['jitter']}, Final delay={info['final_delay']}")
        if show_progress and debug:
            fuzzy_visual_debug(msg_text, stress_score, important, result[1])

    time.sleep(delay)
    return result