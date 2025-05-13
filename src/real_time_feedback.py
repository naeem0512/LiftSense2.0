#!/usr/bin/env python3
#src/real_time_feedback.py
"""
Complete Real-Time Feedback System v4.5

Includes ALL required methods for full operation:
- Data collection and processing
- Physiological monitoring
- AI prediction
- Feedback generation
- Safety controls
- Session management
"""

import numpy as np
import pandas as pd
import time
import os
import joblib
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from src.utils import advanced_recommendation
from src.biosignal_simulator import BioSignalGenerator
import logging
import re

class RealTimeFeedbackSystem:
    def __init__(
        self,
        model_path: str = "data/models/best_lstm_model.keras",
        scaler_path: str = "data/models/scaler.pkl",
        label_encoders: Optional[Dict[str, str]] = None,
        simulation_mode: bool = True,
        safety_mode: bool = True,
        debug: bool = False
    ):
        """Initialize the feedback system with all components"""
        self.debug = debug
        self.safety_mode = safety_mode
        self.simulation_mode = simulation_mode
        
        # Base speed tracking
        self.base_speed = None
        self.base_force = None
        
        # Default user profile
        self.default_profile = {
            'current_weight': 20.0,
            'current_reps': 10,
            'fitness_level': "Intermediate",
            'recovery_rate': 1.2,
            'workout_goal': "Strength",
            'age': 30,
            'gender': "Male",
            'total_sets': 5
        }
        
        # Load AI components
        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            
            default_encoders = {
                'fatigue': "data/models/le_fatigue.pkl",
                'fitness': "data/models/le_fitness.pkl",
                'goal': "data/models/le_goal.pkl",
                'gender': "data/models/le_gender.pkl"
            }
            encoders = default_encoders | (label_encoders or {})
            
            self.le_fatigue = joblib.load(encoders['fatigue'])
            self.le_fitness = joblib.load(encoders['fitness'])
            self.le_goal = joblib.load(encoders['goal']) 
            self.le_gender = joblib.load(encoders['gender'])

        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {str(e)}")

        if simulation_mode:
            self.biosignal_generator = BioSignalGenerator()
        
        # Physiological parameters
        self.rpe_scale = [
            "6-No exertion", "7-Very light", "8", "9-Light",
            "10-Moderate", "11", "12-Somewhat hard", "13",
            "14-Hard", "15-Very hard", "16", "17-Very very hard",
            "18", "19-Extremely hard", "20-Maximal exertion"
        ]
        
        # Session state
        self.session_data = []
        self.current_set = 0
        self.safety_alerts = []
        self.performance_metrics = {
            'velocity_loss': [],
            'fatigue_trend': [],
            'hr_response': [],
            'mechanical_work': []
        }
        
        # Physiological limits
        self.MAX_VELOCITY_LOSS = 40  # Percentage
        self.MIN_GRIP_STRENGTH = 15  # kg
        self.MAX_HR_SAFETY_BUFFER = 5  # bpm below theoretical max
        self.MIN_REP_SPEED = 0.1  # m/s
        self.MAX_REP_SPEED = 2.0  # m/s
        self.MIN_FORCE = 20  # N
        self.MAX_FORCE = 250  # N

    # [Previous methods...]
    # _validate_user_profile, start_session, _print_session_header,
    # _execute_set, _collect_set_data, _display_rep, _update_analytics,
    # _save_crash_report

    def _get_rep_data(self, set_num: int, rep_num: int) -> Dict[str, Any]:
        """Generate realistic rep data with enhanced force calculation"""
        # Initialize base values if not set
        if not hasattr(self, 'base_speed') or self.base_speed is None:
            self.base_speed = 0.8 + (0.1 if self.user_profile['fitness_level'] == "Advanced" else 0)
            self.base_force = 100.0
        
        # Base calculations
        fatigue_factor = (set_num * 0.15) + (rep_num * 0.04)
        fatigue_factor = np.clip(fatigue_factor + np.random.normal(0, 0.02), 0, 0.85)
        
        # Speed calculation with physiological limits
        speed = np.clip(self.base_speed * (1 - fatigue_factor**1.3), 0.1, 2.0)
        
        # Enhanced force calculation with strict limits
        base_force = self.base_force * (1 - fatigue_factor**0.8)
        force = np.clip(base_force, 20, 250)
        
        # Stronger enforcement at low speeds
        if speed < 0.3:
            force = max(force, speed * 80)  # Stronger minimum force
        force = np.clip(force, 20, 250)
        
        # HR response with strict limits
        hr = np.clip(
            65 + (set_num * 8) + (rep_num * 2) + 
            np.random.randint(-3, 4),
            60, self.max_hr
        )
        
        # RPE calculation
        rpe = 8 + int(fatigue_factor * 10)
        
        return {
            "RepSpeed": speed,
            "ForceOutput": force,
            "HeartRate": hr,
            "RPE": rpe
        }

    def _generate_conservative_set(self, user_profile: Dict) -> pd.DataFrame:
        """Generate safe default set data"""
        set_data = []
        for rep_num in range(user_profile.get('current_reps', 10)):
            rep_data = self._get_conservative_rep(self.current_set, rep_num, user_profile)
            set_data.append(rep_data)
        return pd.DataFrame(set_data)

    def _get_conservative_rep(self, set_num: int, rep_num: int, user_profile: Dict) -> Dict:
        """Generate safe default rep data"""
        max_hr = 208 - 0.7 * user_profile['age']
        return {
            "RepSpeed": np.clip(0.8 - (set_num * 0.1) - (rep_num * 0.03), 
                     self.MIN_REP_SPEED, self.MAX_REP_SPEED),
            "ForceOutput": np.clip(80 - (set_num * 8) - (rep_num * 2), 
                         self.MIN_FORCE, self.MAX_FORCE),
            "HeartRate": np.clip(70 + (set_num * 5) + (rep_num * 2), 60, max_hr - 5),
            "GripStrength": np.clip(40 - (set_num * 2) - (rep_num * 0.5), 
                          self.MIN_GRIP_STRENGTH, 50),
            "FitnessLevel": self.le_fitness.transform([user_profile['fitness_level']])[0],
            "RecoveryRate": user_profile['recovery_rate'],
            "WorkoutGoal": self.le_goal.transform([user_profile['workout_goal']])[0],
            "Age": user_profile['age'],
            "Gender": self.le_gender.transform([user_profile['gender']])[0],
            "RPE": min(15, 6 + set_num + rep_num),
            "FatigueFactor": min(0.8, (set_num * 0.15) + (rep_num * 0.05))
        }

    def _validate_biomechanics(self, set_data: pd.DataFrame) -> None:
        """Validate biomechanical parameters with enhanced safety checks"""
        if set_data.empty:
            return
            
        velocity_loss = self._calculate_velocity_loss(set_data)
        if velocity_loss > self.MAX_VELOCITY_LOSS:
            self.safety_alerts.append("EXCESSIVE_FATIGUE")
            print(f"⚠️ Warning: Excessive velocity loss ({velocity_loss:.1f}%)")
            
        fv_corr = set_data["RepSpeed"].corr(set_data["ForceOutput"])
        if fv_corr > -0.2:
            self.safety_alerts.append("ABNORMAL_MECHANICS")
            print(f"⚠️ Warning: Abnormal force-velocity relationship (r={fv_corr:.2f})")
        
        # Add velocity-based force validation
        low_velocity_mask = set_data["RepSpeed"] < 0.3
        if low_velocity_mask.any():
            high_force_mask = set_data["ForceOutput"] > 50
            if (low_velocity_mask & high_force_mask).any():
                self.safety_alerts.append("MECHANICAL_FAULT")
                print("⚠️ Warning: Implausible force at low velocity")
        
        # Add fatigue detection
        if (set_data["RPE"].max() >= 16 and 
            set_data["RepSpeed"].min() < 0.3 and
            "HIGH_FATIGUE" not in self.safety_alerts):
            self.safety_alerts.append("HIGH_FATIGUE")
            print("⚠️ Warning: High fatigue detected")
        
        if (set_data["RepSpeed"].std() < 0.01 or 
            set_data["ForceOutput"].std() < 0.1):
            self.safety_alerts.append("SENSOR_ERROR")
            print("⚠️ Warning: Possible sensor error detected")

    def _prepare_set_data(self, set_data: List[Dict[str, Any]]) -> Dict:
        """Prepares set data for fatigue prediction"""
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(set_data)
        
        # Define required features with default values
        required_numeric = {
            'RepSpeed': 0.0,
            'ForceOutput': 0.0,
            'HeartRate': 70.0,
            'VelocityLoss': 0.0,
            'RecoveryRate': 0.0
        }
        
        required_categorical = {
            'fitness_level': 'Intermediate'  # Default value
        }
        
        # Fill missing numeric columns with defaults
        for col, default in required_numeric.items():
            if col not in df.columns:
                df[col] = default
        
        # Fill missing categorical columns with defaults
        for col, default in required_categorical.items():
            if col not in df.columns:
                df[col] = default
        
        # Extract numeric and categorical features
        numeric_data = df[list(required_numeric.keys())].values
        categorical_data = df[list(required_categorical.keys())].values
        
        # Reshape for LSTM input (samples, timesteps, features)
        numeric_data = numeric_data.reshape(1, numeric_data.shape[0], numeric_data.shape[1])
        categorical_data = categorical_data.reshape(1, categorical_data.shape[0], categorical_data.shape[1])
        
        return {
            "numeric": numeric_data,
            "categorical": categorical_data
        }

    def _predict_fatigue(self, X: np.ndarray) -> Tuple[str, float]:
        """
        Predict fatigue level with proper data handling.
        
        Args:
            X: Input array of shape (1, timesteps, n_features)
            
        Returns:
            Tuple of (predicted_fatigue_level, confidence)
        """
        try:
            # Ensure input is numpy array
            if not isinstance(X, np.ndarray):
                raise ValueError("Input must be numpy array")
            
            # Validate input shape
            if len(X.shape) != 3:
                raise ValueError(f"Expected 3D input (samples, timesteps, features), got shape {X.shape}")
            
            # Flatten temporal data for scaling
            X_flat = X.reshape(-1, X.shape[-1])
            
            # Scale features
            try:
                X_scaled = self.scaler.transform(X_flat)
            except Exception as e:
                logger.error(f"Scaling error: {str(e)}")
                # If scaling fails, try to handle missing features
                if X.shape[-1] != self.scaler.n_features_in_:
                    logger.error(f"Feature mismatch: got {X.shape[-1]} features, expected {self.scaler.n_features_in_}")
                    # Pad or truncate features to match scaler
                    if X.shape[-1] < self.scaler.n_features_in_:
                        X_flat = np.pad(X_flat, ((0, 0), (0, self.scaler.n_features_in_ - X.shape[-1])), mode='constant')
                    else:
                        X_flat = X_flat[:, :self.scaler.n_features_in_]
                    X_scaled = self.scaler.transform(X_flat)
                else:
                    raise
            
            # Reshape back to (samples, timesteps, features)
            X = X_scaled.reshape(X.shape[0], X.shape[1], -1)
            
            # Get prediction
            pred = self.model.predict(X, verbose=0)
            fatigue_idx = np.argmax(pred[0])
            confidence = float(pred[0][fatigue_idx])
            
            # Map to fatigue level
            fatigue_level = self.le_fatigue.inverse_transform([fatigue_idx])[0]
            
            return fatigue_level, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return "Moderate", 0.5  # Safe fallback

    def _parse_recommendation(self, recommendation: str) -> Tuple[float, int, int]:
        """
        Parses the workout recommendation string and extracts weight (kg), reps, and rest (sec).
        Example input: "Next set: 20.0kg, 10 reps, rest for 60 seconds"
        Returns: (weight_kg: float, reps: int, rest_sec: int)
        """
        try:
            # Extract all numeric-like tokens from the recommendation
            numbers = re.findall(r"\d+\.?\d*", recommendation)

            weight_kg = float(numbers[0]) if len(numbers) > 0 else 0.0
            reps = int(numbers[1]) if len(numbers) > 1 else 0
            rest_sec = int(numbers[2]) if len(numbers) > 2 else 0

            return weight_kg, reps, rest_sec
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse recommendation: '{recommendation}' → {e}")
            return 0.0, 0, 0

    def _generate_recommendation(self, fatigue: str, profile: Dict) -> str:
        """Generate workout recommendation with consistent format"""
        try:
            # Get base recommendation
            recommendation = advanced_recommendation(
                fatigue_level=fatigue,
                current_weight=profile['current_weight'],
                current_reps=profile['current_reps'],
                fitness_level=profile['fitness_level'],
                recovery_rate=profile['recovery_rate'],
                workout_goal=profile['workout_goal'],
                age=profile['age'],
                gender=profile['gender'],
                set_number=self.current_set,
                total_sets=profile.get('total_sets', 5)
            )
            
            # Parse recommendation and apply safety limits
            weight_kg, reps, rest_sec = self._parse_recommendation(recommendation)
            
            # Apply safety limits
            weight_kg = max(5.0, min(weight_kg, 250.0))  # 5-250kg range
            reps = max(3, min(reps, 15))  # 3-15 rep range
            rest_sec = max(30, min(rest_sec, 180))  # 30s-3min range
            
            # Generate clean recommendation string with explicit formatting
            return f"Next set: {weight_kg:.1f}kg, {reps} reps, rest for {rest_sec} seconds"
            
        except Exception as e:
            logging.warning(f"Recommendation generation error: {e}")
            # Fallback recommendation with explicit formatting
            return f"Next set: {profile['current_weight']:.1f}kg, {profile['current_reps']} reps, rest for 60 seconds"

    def _calculate_velocity_loss(self, first_rep: Dict[str, Any], last_rep: Dict[str, Any]) -> float:
        """Enhanced velocity loss calculation with edge case handling"""
        first_speed = first_rep["RepSpeed"]
        last_speed = last_rep["RepSpeed"]
        
        # Handle near-zero cases
        if first_speed < 0.3:
            return 100.0 if last_speed < 0.1 else 0.0
        
        # Normal calculation
        velocity_loss = ((first_speed - last_speed) / first_speed) * 100
        
        # Ensure reasonable range
        return np.clip(velocity_loss, 0.0, 100.0)

    def _display_set_summary(self, set_data: List[Dict[str, Any]], fatigue: str, confidence: float, recommendation: str):
        """Display set summary with enhanced metrics"""
        if not set_data:
            return
        
        # Calculate key metrics
        velocity_loss = self._calculate_velocity_loss(set_data[0], set_data[-1])
        avg_hr = np.mean([rep["HeartRate"] for rep in set_data])
        avg_speed = np.mean([rep["RepSpeed"] for rep in set_data])
        avg_force = np.mean([rep["ForceOutput"] for rep in set_data])
        
        print(f"\nSet Summary:")
        print(f"- Fatigue: {fatigue} (Confidence: {confidence:.0%})")
        print(f"- Velocity Loss: {velocity_loss:.1f}%")
        print(f"- Avg HR: {avg_hr:.0f}bpm")
        print(f"- Avg Speed: {avg_speed:.2f}m/s")
        print(f"- Avg Force: {avg_force:.1f}N")
        print(f"- Recommendation: {recommendation}")

    def _log_set_data(self, data: pd.DataFrame, fatigue: str, confidence: float):
        """Record set statistics"""
        self.session_data.append({
            "set": self.current_set,
            "avg_hr": np.mean(data["HeartRate"]),
            "max_hr": np.max(data["HeartRate"]),
            "fatigue": fatigue,
            "confidence": confidence,
            "avg_speed": np.mean(data["RepSpeed"]),
            "avg_rpe": np.mean(data["RPE"]),
            "velocity_loss": self._calculate_velocity_loss(data)
        })

    def _save_session_report(self, profile: Dict):
        """Generate session summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)
        
        with open(report_dir / f"session_{timestamp}.txt", "w") as f:
            f.write("=== Session Report ===\n")
            f.write(f"User: {profile['gender']}, {profile['age']}yo\n")
            f.write(f"Level: {profile['fitness_level']} | Goal: {profile['workout_goal']}\n")
            f.write(f"Recovery Rate: {profile['recovery_rate']:.1f}x\n\n")
            
            f.write("=== Set-by-Set Metrics ===\n")
            for stats in self.session_data:
                f.write(
                    f"Set {stats['set']}:\n"
                    f"  HR: {stats['avg_hr']:.0f}-{stats['max_hr']:.0f}bpm\n"
                    f"  Speed: {stats['avg_speed']:.2f}m/s (Loss: {stats['velocity_loss']:.1f}%)\n"
                    f"  RPE: {stats['avg_rpe']:.1f}\n"
                    f"  Fatigue: {stats['fatigue']} (Confidence: {stats['confidence']:.0%})\n\n"
                )
            
            if self.safety_alerts:
                f.write("\n=== Safety Events ===\n")
                for alert in set(self.safety_alerts):
                    f.write(f"- {alert}\n")
            else:
                f.write("\n=== Safety Events ===\nNo safety events recorded.\n")
            
            f.write("\n=== Performance Trends ===\n")
            f.write(f"Peak Velocity Loss: {max(s['velocity_loss'] for s in self.session_data):.1f}%\n")
            f.write(f"Peak HR: {max(s['max_hr'] for s in self.session_data)}bpm\n")
            f.write(f"Average Mechanical Work: {np.mean(self.performance_metrics['mechanical_work']):.1f} N·m/s\n")
            f.write(f"Fatigue Progression: {', '.join(self.performance_metrics['fatigue_trend'])}\n")
    

    def _save_crash_report(self, profile: Dict, error_msg: str):
        """Save crash report for debugging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"results/crash_{timestamp}.log", "w") as f:
            f.write(f"Crash Report {timestamp}\n")
            f.write(f"Error: {error_msg}\n\n")
            f.write(f"Last Set: {self.current_set}\n")
            f.write(f"Safety Alerts: {self.safety_alerts}\n")
            f.write(f"User Profile: {profile}\n")

    def _handle_emergency(self):
        """Handle critical safety events"""
        print("\n🚨 EMERGENCY STOP!")
        last_alert = self.safety_alerts[-1]
        
        alert_responses = {
            "EMERGENCY_STOP": "Stop immediately and rest for 5 minutes",
            "ARRHYTHMIA_WARNING": "Check pulse and breathing, reduce intensity",
            "EXCESSIVE_FATIGUE": "End session and recover for 48 hours",
            "MECHANICAL_FAULT": "Check equipment and form before continuing",
            "GRIP_FAILURE_WARNING": "Reduce weight and check grip technique",
            "SENSOR_ERROR": "Check sensor connections and recalibrate",
            "ABNORMAL_MECHANICS": "Review exercise technique with trainer"
        }
        
        print(f"Reason: {last_alert}")
        print(f"Action: {alert_responses.get(last_alert, 'Stop and assess')}")
        
        # Log emergency details
        emergency_report = {
            "timestamp": datetime.now().isoformat(),
            "alert": last_alert,
            "set_number": self.current_set,
            "recommended_action": alert_responses.get(last_alert),
            "physiological_state": {
                "last_hr": self.session_data[-1]['max_hr'] if self.session_data else None,
                "last_fatigue": self.session_data[-1]['fatigue'] if self.session_data else None
            }
        }
        
        emergency_dir = Path("results/emergencies")
        emergency_dir.mkdir(exist_ok=True)
        with open(emergency_dir / f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(emergency_report, f, indent=2)

    def _validate_profile(self, profile: Dict) -> None:
        """Validate user profile with enhanced checks"""
        required_keys = [
            'current_weight', 'current_reps', 'fitness_level',
            'recovery_rate', 'workout_goal', 'age', 'gender'
        ]
        
        # Check for missing keys
        missing = [k for k in required_keys if k not in profile]
        if missing:
            raise ValueError(f"Profile missing required keys: {missing}")
        
        # Validate numeric ranges
        if not (0.5 <= profile['recovery_rate'] <= 1.5):
            raise ValueError("Invalid recovery rate (0.5-1.5 expected)")
        
        if not (15 <= profile['age'] <= 80):
            raise ValueError("Age outside valid range (15-80)")
        
        if not (5.0 <= profile['current_weight'] <= 500.0):
            raise ValueError("Weight outside valid range (5-500kg)")
        
        if not (3 <= profile['current_reps'] <= 30):
            raise ValueError("Reps outside valid range (3-30)")
        
        # Validate categorical values
        valid_fitness = ["Beginner", "Intermediate", "Advanced"]
        if profile['fitness_level'] not in valid_fitness:
            raise ValueError(f"Invalid fitness level. Must be one of {valid_fitness}")
        
        valid_goals = ["Strength", "Endurance", "Hypertrophy"]
        if profile['workout_goal'] not in valid_goals:
            raise ValueError(f"Invalid workout goal. Must be one of {valid_goals}")
        
        valid_genders = ["Male", "Female", "Other"]
        if profile['gender'] not in valid_genders:
            raise ValueError(f"Invalid gender. Must be one of {valid_genders}")

    def start_session(self, user_profile: Dict = None) -> None:
        """Execute full workout session with enhanced error handling"""
        try:
            # Use default profile if none provided
            if user_profile is None:
                user_profile = self.default_profile
                print("Using default profile")
            
            # Validate profile
            self._validate_profile(user_profile)
            
            # Store profile for session
            self.user_profile = user_profile
            self.max_hr = 208 - 0.7 * user_profile['age']
            
            print("\n===================== WORKOUT SESSION ======================")
            print(f"User: {user_profile['gender']}, {user_profile['age']}yo (Max HR: {self.max_hr:.1f}bpm)")
            print(f"Level: {user_profile['fitness_level']} | Goal: {user_profile['workout_goal']}")
            print(f"Recovery Rate: {user_profile['recovery_rate']:.1f}x | Safety: ON")
            print("============================================================\n")
            
            current_weight = user_profile['current_weight']
            current_reps = user_profile['current_reps']
            
            for set_num in range(user_profile['total_sets']):
                try:
                    print(f"\n▶ Set {set_num + 1} (Weight: {current_weight:.1f}kg)")
                    self.current_set = []
                    self.safety_alerts = []
                    
                    # Reset base speed for new set
                    self.base_speed = None
                    self.base_force = None
                    
                    for rep_num in range(current_reps):
                        rep_data = self._get_rep_data(set_num, rep_num)
                        
                        # Store first rep as base
                        if rep_num == 0:
                            self.base_speed = rep_data["RepSpeed"]
                            self.base_force = rep_data["ForceOutput"]
                            print(f"  Rep {rep_num + 1}: HR={rep_data['HeartRate']:3.0f}bpm | "
                                  f"Speed={rep_data['RepSpeed']:.2f}m/s (Base) | "
                                  f"Force={rep_data['ForceOutput']:6.1f}N | "
                                  f"RPE={rep_data['RPE']}")
                        else:
                            print(f"  Rep {rep_num + 1}: HR={rep_data['HeartRate']:3.0f}bpm | "
                                  f"Speed={rep_data['RepSpeed']:.2f}m/s | "
                                  f"Force={rep_data['ForceOutput']:6.1f}N | "
                                  f"RPE={rep_data['RPE']}")
                        
                        if not self._validate_rep(rep_data, set_num, rep_num):
                            break
                            
                        self.current_set.append(rep_data)
                    
                    # Process set and get recommendations
                    fatigue_level, confidence = self._predict_fatigue(np.array([self.current_set]))
                    recommendation = self._generate_recommendation(fatigue_level, user_profile)
                    print(f"\n{recommendation}")
                    
                    # Update for next set
                    current_weight = float(recommendation.split()[2].replace('kg', ''))
                    current_reps = int(recommendation.split()[4])
                    
                except Exception as e:
                    print(f"⚠️ Continuing session with reduced intensity: {str(e)}")
                    current_weight *= 0.8
                    continue
            
            print("\nSession completed successfully")
            
        except Exception as e:
            print(f"⚠️ Session recovery: {str(e)}")
            if user_profile != self.default_profile:
                print("Attempting session with default profile...")
                self.start_session()  # Restart with default profile
            else:
                raise RuntimeError("Session failed with both custom and default profiles")

    def _print_session_header(self, profile: Dict):
        """Display session info with physiological limits"""
        max_hr = 208 - 0.7 * profile['age']
        print(f"\n{' WORKOUT SESSION ':=^60}")
        print(f"User: {profile['gender']}, {profile['age']}yo (Max HR: {max_hr}bpm)")
        print(f"Level: {profile['fitness_level']} | Goal: {profile['workout_goal']}")
        print(f"Recovery Rate: {profile['recovery_rate']:.1f}x | Safety: {'ON' if self.safety_mode else 'OFF'}")
        print("=" * 60)

    def _execute_set(self, user_profile: Dict) -> Optional[Dict]:
        """Enhanced set execution with comprehensive monitoring"""
        print(f"\n▶ Set {self.current_set} (Weight: {user_profile['current_weight']}kg)")
        
        try:
            # Data collection with timeout
            set_data = self._collect_set_data(user_profile)
            if set_data is None:
                print("⚠️ Data collection failed, using conservative defaults")
                set_data = self._generate_conservative_set(user_profile)
            
            # Biomechanical validation
            if self.safety_mode:
                self._validate_biomechanics(set_data)
            
            # AI prediction with confidence
            fatigue, confidence = self._predict_fatigue(set_data.values.reshape(1, -1, set_data.shape[-1]))
            
            # Generate science-based recommendation
            recommendation = self._generate_recommendation(fatigue, user_profile)
            adjusted_params = self._parse_recommendation(recommendation)
            
            # Update user profile for next set
            user_profile.update(adjusted_params)
            
            # Log and display results
            self._log_set_data(set_data, fatigue, confidence)
            self._display_set_summary(set_data, fatigue, confidence, recommendation)
            
            return {
                'fatigue': fatigue,
                'confidence': f"{confidence:.0%}",
                'recommendation': recommendation,
                'metrics': {
                    'avg_speed': np.mean(set_data["RepSpeed"]),
                    'velocity_loss': self._calculate_velocity_loss(set_data)
                }
            }
            
        except Exception as e:
            print(f"🚨 Set processing error: {str(e)}")
            return None

    def _collect_set_data(self, user_profile: Dict) -> Optional[pd.DataFrame]:
        """Collect metrics for all reps in a set"""
        set_data = []
        intensity = self.current_set / user_profile.get('total_sets', 5)
        
        for rep_num in range(user_profile.get('current_reps', 10)):
            rep_data = self._get_rep_data(
                set_num=self.current_set,
                rep_num=rep_num,
                user_profile=user_profile,
                intensity=intensity
            )
            
            if rep_data is None:
                rep_data = self._get_conservative_rep(self.current_set, rep_num, user_profile)
                
            if self.safety_mode:
                self._validate_rep(rep_data, self.current_set, rep_num)
                if "EMERGENCY_STOP" in self.safety_alerts:
                    return None
                    
            set_data.append(rep_data)
            self._display_rep(rep_num, rep_data)
            time.sleep(0.2)  # Simulate rep duration
            
        return pd.DataFrame(set_data)

    def _update_analytics(self, set_result: Dict):
        """Update performance metrics"""
        if set_result:
            self.performance_metrics['fatigue_trend'].append(set_result['fatigue'])
            self.performance_metrics['velocity_loss'].append(set_result['metrics']['velocity_loss'])
            self.performance_metrics['hr_response'].append(set_result['metrics']['avg_hr'])
            self.performance_metrics['mechanical_work'].append(
                set_result['metrics']['avg_speed'] * 
                set_result['metrics']['force_output']
            )

    def _validate_rep(self, rep_data: Dict[str, Any], set_num: int, rep_num: int) -> bool:
        """Enhanced rep validation with biomechanical checks"""
        try:
            # Basic range validation
            if rep_data["RepSpeed"] < self.MIN_REP_SPEED or rep_data["RepSpeed"] > self.MAX_REP_SPEED:
                self.safety_alerts.append("INVALID_SPEED")
                logging.warning(f"Invalid speed: {rep_data['RepSpeed']:.2f} m/s")
                return False
            
            if rep_data["ForceOutput"] < self.MIN_FORCE or rep_data["ForceOutput"] > self.MAX_FORCE:
                self.safety_alerts.append("INVALID_FORCE")
                logging.warning(f"Invalid force: {rep_data['ForceOutput']:.1f} N")
                return False
            
            # Force-velocity relationship validation
            if rep_data["RepSpeed"] < 0.3 and rep_data["ForceOutput"] > rep_data["RepSpeed"] * 200:
                self.safety_alerts.append("ABNORMAL_FV")
                logging.warning("Abnormal force-velocity relationship")
                return False
            
            # HR validation
            max_hr = 208 - 0.7 * self.user_profile['age']
            if rep_data["HeartRate"] > max_hr - self.MAX_HR_SAFETY_BUFFER:
                self.safety_alerts.append("HIGH_HR")
                logging.warning(f"High heart rate: {rep_data['HeartRate']} bpm")
                return False
            
            # Velocity loss validation
            if rep_num > 0:
                velocity_loss = (self.base_speed - rep_data["RepSpeed"]) / self.base_speed * 100
                if velocity_loss > self.MAX_VELOCITY_LOSS:
                    self.safety_alerts.append("EXCESSIVE_VELOCITY_LOSS")
                    logging.warning(f"Excessive velocity loss: {velocity_loss:.1f}%")
                    return False
                
            # RPE validation
            if rep_data["RPE"] > 15:
                self.safety_alerts.append("HIGH_RPE")
                logging.warning(f"High RPE: {rep_data['RPE']}")
                return False
            
            return True
        
        except Exception as e:
            logging.error(f"Rep validation error: {str(e)}")
            return False

    def _display_rep(self, rep_num: int, data: Dict):
        """Enhanced rep display with color coding"""
        rpe_idx = min(len(self.rpe_scale)-1, int(data['RPE'])-6)
        rpe_desc = self.rpe_scale[rpe_idx]
        
        # Color coding based on intensity
        if data['RPE'] >= 14:
            intensity = "\033[91m"  # Red for high intensity
        elif data['RPE'] >= 10:
            intensity = "\033[93m"  # Yellow for moderate
        else:
            intensity = "\033[92m"  # Green for low
            
        print(
            f"  Rep {rep_num + 1}: "
            f"HR={data['HeartRate']:3d}bpm | "
            f"Speed={intensity}{data['RepSpeed']:.2f}m/s\033[0m | "
            f"Force={data['ForceOutput']:5.1f}N | "
            f"RPE={rpe_desc}"
        )

if __name__ == "__main__":
    # Example usage with comprehensive profile
    feedback = RealTimeFeedbackSystem(
        simulation_mode=True,
        safety_mode=True,
        debug=True
    )

    feedback.start_session({
        'current_weight': 20.0,    # kg
        'current_reps': 10,
        'fitness_level': "Intermediate",
        'recovery_rate': 1.2,
        'workout_goal': "Strength",
        'age': 30,
        'gender': "Male",
        'total_sets': 5
    })