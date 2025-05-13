#!/usr/bin/env python3
"""
LiftSense 2.0 - Enhanced Feedback System with RL Integration

Extends the EnhancedFeedbackSystem with reinforcement learning-based
recommendations for more intelligent and personalized workout guidance.
"""

import numpy as np
import pandas as pd
import time
import logging
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from src.enhanced_feedback import EnhancedFeedbackSystem
from src.rl_recommendation import ReinforcementLearningRecommender


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class RLEnhancedFeedbackSystem(EnhancedFeedbackSystem):
    """
    Enhanced feedback system with reinforcement learning capabilities
    
    Integrates a RL-based recommendation system with the existing
    LiftSense feedback system for more intelligent workout recommendations.
    """
    
    def __init__(
        self,
        model_path="data/models/lstm_model.keras",
        scaler_path="data/models/scaler.pkl",
        rl_model_path="data/models/rl",
        use_rl=True,
        train_rl_if_needed=True,
        simulation_mode=False,
        safety_mode=True,
        safety_weight: float = 0.5,
        performance_weight: float = 0.3,
        adherence_weight: float = 0.2,
        **kwargs
    ):
        """Initialize RL-enhanced feedback system
        
        Args:
            model_path: Path to base model
            scaler_path: Path to scaler
            rl_model_path: Path to RL model directory
            use_rl: Whether to use RL recommendations
            train_rl_if_needed: Whether to train RL model if not found
            simulation_mode: Whether to run in simulation mode
            safety_mode: Whether to use safety-enhanced models
            safety_weight: Weight for safety objective in RL
            performance_weight: Weight for performance objective
            adherence_weight: Weight for adherence objective
            **kwargs: Additional arguments for parent class
        """
        # Initialize parent class
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            ensemble_path="data/models/v2.0/ensemble",
            use_safety_models=safety_mode,
            **kwargs
        )
        
        # RL configuration
        self.use_rl = use_rl
        self.train_rl_if_needed = train_rl_if_needed
        self.simulation_mode = simulation_mode
        self.rl_model_path = Path(rl_model_path)
        self.safety_weight = safety_weight
        self.performance_weight = performance_weight
        self.adherence_weight = adherence_weight
        
        # Initialize RL recommender
        self.rl_recommender = None
        self.is_trained = False  # Track RL model training status
        
        # Initialize debug info
        self.debug_info = {
            'successful_rl_recommendations': 0,
            'fallback_recommendations': 0,
            'rl_errors': 0,
            'safety_events': 0
        }
        
        # Load RL model if available
        if self.use_rl:
            try:
                from src.rl_recommendation import RLRecommender
                self.rl_recommender = RLRecommender(
                    model_dir=str(self.rl_model_path),
                    simulation_mode=simulation_mode
                )
                logging.info("RL Recommender initialized")
                
                # Try to load existing model
                model_path = self.rl_model_path / "rl_model_final.weights.h5"
                if model_path.exists():
                    self.rl_recommender.load_model(str(model_path))
                    self.is_trained = True
                    logging.info(f"Loaded RL model from {model_path}")
                elif self.train_rl_if_needed:
                    logging.info("No RL model found, will train if needed")
                else:
                    logging.warning("No RL model found and training disabled")
                    
            except Exception as e:
                logging.error(f"Error initializing RL recommender: {e}")
                self.rl_recommender = None
        
        # Recommendation tracking for performance analysis
        self.rl_recommendations = []
        self.traditional_recommendations = []
        
        # State tracking for RL
        self.current_state = None
        
        # Fix model input shapes
        self._fix_model_input_shapes()
    
    def _parse_recommendation(self, recommendation_text):
        """
        Safely parse a recommendation string
        
        Args:
            recommendation_text: Text of recommendation
            
        Returns:
            Tuple of (weight, reps, rest_time) or None if parsing fails
        """
        try:
            # Match weight and reps from recommendation text
            match = re.search(r"Next set: (\d+\.?\d*)\s*(?:kg)?[,.]?\s*(\d+) reps", recommendation_text)
            if not match:
                return None
                
            weight_str = match.group(1).strip().replace(',', '.')  # Replace any commas with periods
            weight = float(weight_str)
            reps = int(match.group(2))
            
            # Look for rest time
            rest_match = re.search(r"rest for (\d+) seconds", recommendation_text)
            rest_time = int(rest_match.group(1)) if rest_match else 60
            
            return (weight, reps, rest_time)
        except (ValueError, IndexError, TypeError) as e:
            logging.warning(f"Error parsing recommendation: {e}")
            return None

    def get_recommendation(self, user_profile, fatigue_level=None, velocity_loss=None, fatigue_confidence=0.5):
        """
        Get workout recommendation based on user profile using RL policy
        
        Args:
            user_profile: Dictionary with user data
            fatigue_level: Fatigue prediction from LiftSense
            velocity_loss: Velocity loss percentage from current set
            fatigue_confidence: Confidence in fatigue prediction
            
        Returns:
            Dictionary with weight, reps, and rest time recommendations
        """
        if not self.is_trained:
            logging.warning("RL model not trained or loaded. Attempting to load default model.")
            if not self.load_model():
                logging.error("Failed to load RL model. Using fallback recommendation.")
                return self._fallback_recommendation(user_profile, fatigue_level)
        
        try:
            # Convert user profile to state vector
            state_vector = self.map_user_profile_to_state(
                user_profile, 
                fatigue_level=fatigue_level,
                fatigue_confidence=fatigue_confidence,
                velocity_loss=velocity_loss
            )
            
            # Reshape for model input
            state = np.reshape(state_vector, [1, self.state_size])
            
            # Get best action from RL agent
            action = self.agent.act(state, training=False)
            
            # Decode action
            weight_idx = action // (len(self.env.rep_actions) * len(self.env.rest_actions))
            remaining = action % (len(self.env.rep_actions) * len(self.env.rest_actions))
            rep_idx = remaining // len(self.env.rest_actions)
            rest_idx = remaining % len(self.env.rest_actions)
            
            weight_change = self.env.weight_actions[weight_idx]
            rep_change = self.env.rep_actions[rep_idx]
            rest_change = self.env.rest_actions[rest_idx]
            
            # Apply recommendations to user profile
            current_weight = user_profile.get('current_weight', 20.0)
            current_reps = user_profile.get('current_reps', 10)
            current_rest = user_profile.get('rest_time', 60)
            
            # Calculate new values with physiological constraints
            new_weight = round(current_weight * (1 + weight_change), 1)
            new_weight = max(5.0, min(250.0, new_weight))  # Safety bounds
            
            new_reps = max(5, min(20, int(current_reps + rep_change)))
            new_rest = max(30, min(180, int(current_rest + rest_change)))
            
            # Take action in environment to get feedback and confidence
            # This does not modify the environment state, just simulates the outcome
            test_state = state_vector.copy() if state_vector is not None else np.zeros(self.state_size)
            next_state, reward, _, info = self.env.step(action)
            
            # Adjust confidence based on reward
            confidence = min(0.9, max(0.1, (reward + 3) / 6))  # Map reward (-3 to +3) to confidence (0.1 to 0.9)
            
            # Create recommendation object with detailed feedback
            recommendation = {
                'weight': new_weight,
                'reps': new_reps,
                'rest_time': new_rest,
                'confidence': confidence,
                'safety_score': info.get('safety_reward', 0.0),
                'performance_score': info.get('performance_reward', 0.0),
                'adherence_score': info.get('adherence_reward', 0.0),
                'reward': reward
            }
            
            return recommendation
            
        except Exception as e:
            logging.error(f"Error generating RL recommendation: {e}")
            return self._fallback_recommendation(user_profile, fatigue_level)

    def map_user_profile_to_state(self, user_profile, fatigue_level=None, fatigue_confidence=0.5, velocity_loss=None):
        """
        Map LiftSense user profile to RL state vector
        
        Args:
            user_profile: Dictionary with user data
            fatigue_level: Optional fatigue level from prediction
            fatigue_confidence: Confidence in fatigue prediction
            velocity_loss: Optional velocity loss percentage
            
        Returns:
            Numpy array state vector
        """
        if user_profile is None:
            logging.warning("User profile is None, using defaults")
            return np.zeros(self.state_size)
            
        state = np.zeros(self.state_size)
        
        try:
            # Map weight to percentage of estimated 1RM
            current_weight = user_profile.get('current_weight', 50)
            if user_profile.get('fitness_level') == "Beginner":
                estimated_1rm = current_weight * 1.5
            elif user_profile.get('fitness_level') == "Advanced":
                estimated_1rm = current_weight * 1.3
            else:
                estimated_1rm = current_weight * 1.4
            
            weight_pct = min(0.95, current_weight / max(1, estimated_1rm))
            state[0] = weight_pct
            
            # Map reps directly
            state[1] = user_profile.get('current_reps', 10)
            
            # Map fatigue level
            if fatigue_level:
                if fatigue_level == "High":
                    state[2] = 0.8 * fatigue_confidence + 0.6 * (1 - fatigue_confidence)
                elif fatigue_level == "Moderate":
                    state[2] = 0.5 * fatigue_confidence + 0.3 * (1 - fatigue_confidence)
                else:  # Low
                    state[2] = 0.2 * fatigue_confidence + 0.3 * (1 - fatigue_confidence)
            else:
                # Default moderate fatigue if no prediction
                state[2] = 0.4
            
            # Map fitness level to 0-1 scale
            fitness = user_profile.get('fitness_level', 'Intermediate')
            if fitness == "Beginner":
                state[3] = 0.3
            elif fitness == "Advanced":
                state[3] = 0.8
            else:  # Intermediate
                state[3] = 0.6
            
            # Map recovery rate directly
            state[4] = user_profile.get('recovery_rate', 1.0)
            
            # Map workout goal
            goal = user_profile.get('workout_goal', 'Strength')
            if goal == "Strength":
                state[5] = 0.0
            elif goal == "Hypertrophy":
                state[5] = 0.5
            else:  # Endurance
                state[5] = 1.0
            
            # Map age to normalized factor (younger = higher value)
            age = user_profile.get('age', 30)
            state[6] = max(0.1, min(0.9, 1.0 - (age - 20) / 60))
            
            # Map gender
            gender = user_profile.get('gender', 'Male')
            state[7] = 1.0 if gender == "Male" else 0.0
            
            # Map set number (normalized)
            set_num = user_profile.get('set_number', 0)
            total_sets = user_profile.get('total_sets', 5)
            state[8] = min(1.0, set_num / max(1, total_sets))
            
            # Map velocity loss
            if velocity_loss is not None:
                state[9] = min(1.0, velocity_loss / 100.0)  # Normalize 0-100% to 0-1
            else:
                # Estimate from fatigue if not provided
                state[9] = state[2] * 1.2  # Velocity loss correlates with fatigue
            
            return state
        except Exception as e:
            logging.error(f"Error mapping user profile to state: {e}")
            return np.zeros(self.state_size)

    def _generate_recommendation(self, fatigue_level: str, user_profile: Dict) -> str:
        """
        Override the parent method to add enhanced RL recommendations
        
        Args:
            fatigue_level: Predicted fatigue level
            user_profile: User profile data
            
        Returns:
            Recommendation string
        """
        # First get the basic recommendation from parent
        basic_rec = super()._generate_recommendation(fatigue_level, user_profile)
        
        if not hasattr(self, 'traditional_recommendations'):
            self.traditional_recommendations = []
        self.traditional_recommendations.append(basic_rec)
        
        # If RL is disabled or RL recommender not available, use traditional approach
        if not self.use_rl or not hasattr(self, 'rl_recommender') or self.rl_recommender is None:
            return basic_rec
        
        # Use the safe recommendation function
        return self.get_safe_recommendation(user_profile, fatigue_level)
    
    def _predict_fatigue(self, set_data):
        """Override to use ensemble prediction with padding when available"""
        if not self.use_ensemble:
            # Fallback to standard model
            return super()._predict_fatigue(set_data)
        
        try:
            logging.info("Using ensemble for prediction")
            # Prepare data
            prepared_data = super()._prepare_set_data(set_data)
            X = prepared_data["numeric"]
            
            # Pad input if needed for safety model
            X = self._pad_input_for_safety_model(X)
            
            # Get prediction from ensemble
            pred_classes, confidence, fatigue_labels = self.ensemble.predict(X)
            
            # Store confidence for RL
            self.last_confidence = confidence[0]
            
            return fatigue_labels[0], confidence[0]
        except Exception as e:
            logging.error(f"Ensemble prediction error: {e}")
            # Fallback to standard model
            logging.info("Falling back to standard model")
            return super()._predict_fatigue(set_data)
    
    def _execute_set(self, user_profile: Dict) -> Optional[Dict]:
        """Execute a set with enhanced RL recommendation tracking"""
        # Keep track of set number for RL
        if not hasattr(self, 'current_set_num'):
            self.current_set_num = 0
        else:
            self.current_set_num += 1
            
        # Execute the set using parent implementation
        result = super()._execute_set(user_profile)
        
        # Store results for RL learning
        if result and self.use_rl and hasattr(self, 'current_set'):
            try:
                # Capture state transition for potential online learning
                velocity_loss = self._calculate_velocity_loss(
                    self.current_set[0], self.current_set[-1]
                ) if len(self.current_set) > 1 else 0
                
                self.current_state = {
                    'user_profile': user_profile.copy(),
                    'fatigue_level': result['fatigue'],
                    'fatigue_confidence': float(result['confidence'].strip('%')) / 100,
                    'velocity_loss': velocity_loss,
                    'heart_rate': np.mean([rep.get('HeartRate', 70) for rep in self.current_set]),
                    'rep_speed': np.mean([rep.get('RepSpeed', 0.8) for rep in self.current_set]),
                    'force_output': np.mean([rep.get('ForceOutput', 100) for rep in self.current_set]),
                    'set_number': self.current_set_num
                }
            except Exception as e:
                logging.warning(f"Error capturing RL state: {e}")
                
        return result
    
    def analyze_rl_performance(self) -> Dict:
        """
        Analyze the performance comparison between RL and traditional recommendations
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.rl_recommendations:
            return {"error": "No RL recommendations recorded"}
            
        try:
            # Extract values from both recommendation types
            trad_weights = []
            trad_reps = []
            trad_rest = []
            
            rl_weights = []
            rl_reps = []
            rl_rest = []
            rl_confidences = []
            rl_safety_scores = []
            
            # Parse traditional recommendations
            for rec in self.traditional_recommendations:
                parsed = self._parse_recommendation(rec)
                if parsed:
                    weight, reps, rest = parsed
                    trad_weights.append(weight)
                    trad_reps.append(reps)
                    trad_rest.append(rest)
            
            # Extract RL recommendations
            for rec in self.rl_recommendations:
                data = rec.get('data', {})
                rl_weights.append(data.get('weight', 0))
                rl_reps.append(data.get('reps', 0))
                rl_rest.append(data.get('rest_time', 0))
                rl_confidences.append(data.get('confidence', 0))
                rl_safety_scores.append(data.get('safety_score', 0))
            
            # Calculate comparison metrics
            metrics = {
                'count': len(self.rl_recommendations),
                'weight_difference': np.mean(np.array(rl_weights) - np.array(trad_weights)) if trad_weights and rl_weights else 0,
                'rep_difference': np.mean(np.array(rl_reps) - np.array(trad_reps)) if trad_reps and rl_reps else 0,
                'rest_difference': np.mean(np.array(rl_rest) - np.array(trad_rest)) if trad_rest and rl_rest else 0,
                'rl_avg_confidence': np.mean(rl_confidences) if rl_confidences else 0,
                'rl_avg_safety': np.mean(rl_safety_scores) if rl_safety_scores else 0
            }
            
            # Summarize findings
            analysis = {
                'metrics': metrics,
                'summary': {
                    'more_conservative': metrics['weight_difference'] < -0.5 or metrics['rep_difference'] < -0.5,
                    'higher_rest': metrics['rest_difference'] > 5,
                    'confidence': "High" if metrics['rl_avg_confidence'] > 0.7 else "Moderate"
                }
            }
            
            logging.info(f"RL Performance Analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing RL performance: {e}")
            return {"error": str(e)}
    
    def save_session_report(self, profile: Dict) -> str:
        """
        Enhanced session report with RL performance analysis
        
        Args:
            profile: User profile
            
        Returns:
            Path to saved report
        """
        # Get base report path from parent
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"session_{timestamp}.txt"
        
        # Add RL analysis to report
        with open(report_path, "w") as f:
            f.write("=== Session Report ===\n")
            f.write(f"User: {profile['gender']}, {profile['age']}yo\n")
            f.write(f"Level: {profile['fitness_level']} | Goal: {profile['workout_goal']}\n")
            f.write(f"Recovery Rate: {profile['recovery_rate']:.1f}x\n\n")
            
            f.write("=== Set-by-Set Metrics ===\n")
            for stats in self.session_data:
                f.write(
                    f"Set {stats['set'] + 1}:\n"
                    f"  HR: {stats['avg_hr']:.0f}-{stats['max_hr']:.0f}bpm\n"
                    f"  Speed: {stats['avg_speed']:.2f}m/s (Loss: {stats['velocity_loss']:.1f}%)\n"
                    f"  Fatigue: {stats['fatigue']} (Confidence: {stats['confidence']:.0%})\n"
                )
                
                # Add RL recommendation if available
                if hasattr(self, 'rl_recommendations') and len(self.rl_recommendations) > stats['set']:
                    rl_rec = self.rl_recommendations[stats['set']]['formatted']
                    f.write(f"  RL Recommendation: {rl_rec}\n")
                
                # Add traditional recommendation if available
                if hasattr(self, 'traditional_recommendations') and len(self.traditional_recommendations) > stats['set']:
                    trad_rec = self.traditional_recommendations[stats['set']]
                    f.write(f"  Traditional Recommendation: {trad_rec}\n\n")
                else:
                    f.write("\n")
            
            # Add RL performance analysis
            if hasattr(self, 'use_rl') and self.use_rl:
                rl_analysis = self.analyze_rl_performance()
                if "error" not in rl_analysis:
                    f.write("\n=== RL Recommendation Analysis ===\n")
                    metrics = rl_analysis['metrics']
                    f.write(f"Recommendations compared: {metrics['count']}\n")
                    f.write(f"RL vs Traditional differences:\n")
                    f.write(f"  Weight: {metrics['weight_difference']:.1f}kg\n")
                    f.write(f"  Reps: {metrics['rep_difference']:.1f}\n")
                    f.write(f"  Rest: {metrics['rest_difference']:.1f}s\n")
                    f.write(f"Average RL confidence: {metrics['rl_avg_confidence']:.2f}\n")
                    f.write(f"Average RL safety score: {metrics['rl_avg_safety']:.2f}\n")
                    
                    summary = rl_analysis['summary']
                    f.write("\nSummary: ")
                    if summary['more_conservative']:
                        f.write("RL recommendations were more conservative (lower weight/reps). ")
                    if summary['higher_rest']:
                        f.write("RL suggested longer rest periods. ")
                    f.write(f"RL confidence was {summary['confidence'].lower()}.\n")
                
            # Add safety events if any
            if self.safety_alerts:
                f.write("\n=== Safety Events ===\n")
                for alert in set(self.safety_alerts):
                    f.write(f"- {alert}\n")
            else:
                f.write("\n=== Safety Events ===\nNo safety events recorded.\n")
            
            # Add performance trends
            f.write("\n=== Performance Trends ===\n")
            f.write(f"Peak Velocity Loss: {max([s['velocity_loss'] for s in self.session_data], default=0):.1f}%\n")
            f.write(f"Peak HR: {max([s['max_hr'] for s in self.session_data], default=0)}bpm\n")
            
            if hasattr(self, 'performance_metrics') and 'mechanical_work' in self.performance_metrics:
                f.write(f"Average Mechanical Work: {np.mean(self.performance_metrics['mechanical_work']):.1f} N·m/s\n")
            
            if hasattr(self, 'performance_metrics') and 'fatigue_trend' in self.performance_metrics:
                f.write(f"Fatigue Progression: {', '.join(self.performance_metrics['fatigue_trend'])}\n")
        
        # Save RL analysis as JSON for further analysis
        if hasattr(self, 'use_rl') and self.use_rl and hasattr(self, 'rl_recommendations') and len(self.rl_recommendations) > 0:
            try:
                # Convert RL recommendations to serializable format
                serializable_recs = []
                for rec in self.rl_recommendations:
                    serializable_rec = {
                        'formatted': rec.get('formatted', ''),
                        'fatigue_level': rec.get('fatigue_level', ''),
                        'data': {}
                    }
                    
                    # Convert data dictionary if it exists
                    if 'data' in rec and rec['data']:
                        for key, value in rec['data'].items():
                            if isinstance(value, (np.integer, np.floating, np.bool_)):
                                serializable_rec['data'][key] = float(value) if isinstance(value, np.floating) else (
                                    int(value) if isinstance(value, np.integer) else bool(value)
                                )
                            else:
                                serializable_rec['data'][key] = value
                                
                    serializable_recs.append(serializable_rec)
                
                # Prepare analysis for serialization
                rl_analysis = self.analyze_rl_performance()
                serializable_analysis = {}
                
                if "error" not in rl_analysis:
                    # Convert metrics to serializable format
                    metrics = {}
                    for key, value in rl_analysis.get('metrics', {}).items():
                        if isinstance(value, (np.integer, np.floating, np.bool_)):
                            metrics[key] = float(value) if isinstance(value, np.floating) else (
                                int(value) if isinstance(value, np.integer) else bool(value)
                            )
                        else:
                            metrics[key] = value
                    
                    # Convert summary to serializable format
                    summary = {}
                    for key, value in rl_analysis.get('summary', {}).items():
                        if isinstance(value, (np.integer, np.floating, np.bool_)):
                            summary[key] = float(value) if isinstance(value, np.floating) else (
                                int(value) if isinstance(value, np.integer) else bool(value)
                            )
                        else:
                            summary[key] = value
                    
                    serializable_analysis = {
                        'metrics': metrics,
                        'summary': summary
                    }
                else:
                    serializable_analysis = {'error': rl_analysis.get('error')}
                
                # Create final serializable results
                rl_results = {
                    "recommendations": serializable_recs,
                    "traditional": self.traditional_recommendations if hasattr(self, 'traditional_recommendations') else [],
                    "analysis": serializable_analysis
                }
                
                # Write to JSON file using custom encoder to handle NumPy types
                with open(report_dir / f"rl_analysis_{timestamp}.json", "w") as f:
                    json.dump(rl_results, f, indent=2, cls=NumpyEncoder)
                    
            except Exception as e:
                logging.error(f"Error saving RL analysis: {e}")
        
        return str(report_path)
    
    def suggest_rl_training_strategy(self) -> Dict:
        """
        Suggest a strategy for improving the RL model based on session data
        
        Returns:
            Dictionary with training strategy suggestions
        """
        # Default suggestions if no session data
        default_suggestions = ["Collect more workout data before training RL model"]
        
        # Need at least a few sets of data to make suggestions
        if not hasattr(self, 'session_data') or len(self.session_data) < 1:
            return {
                "suggestions": default_suggestions,
                "recommended_weights": {
                    "safety_weight": self.safety_weight,
                    "performance_weight": self.performance_weight,
                    "adherence_weight": self.adherence_weight
                },
                "current_weights": {
                    "safety_weight": self.safety_weight,
                    "performance_weight": self.performance_weight,
                    "adherence_weight": self.adherence_weight
                }
            }
        
        # Analyze the session data for patterns
        fatigue_changes = []
        previous_fatigue = None
        
        # Initialize with default suggestions
        suggestions = []
        
        try:
            # Process session data
            for stats in self.session_data:
                fatigue = stats.get('fatigue', 'Moderate')
                if previous_fatigue:
                    if fatigue == "High" and previous_fatigue != "High":
                        fatigue_changes.append("Increased to High")
                    elif fatigue == "Low" and previous_fatigue == "High":
                        fatigue_changes.append("Decreased to Low")
                previous_fatigue = fatigue
            
            # Look for safety events
            safety_issues = hasattr(self, 'safety_alerts') and len(self.safety_alerts) > 0
            
            # Check for rapid fatigue changes
            if "Increased to High" in fatigue_changes:
                suggestions.append("Increase safety_weight to make RL more conservative before high fatigue")
            
            # Check for safety issues
            if safety_issues:
                suggestions.append("Increase safety_weight and add specific rewards for avoiding detected safety issues")
            
            # Check fatigue distribution
            if hasattr(self, 'session_data'):
                fatigue_counts = {"Low": 0, "Moderate": 0, "High": 0}
                for stats in self.session_data:
                    fatigue = stats.get('fatigue', 'Moderate')
                    fatigue_counts[fatigue] += 1
                
                # Check for imbalanced fatigue levels
                if fatigue_counts["High"] == 0:
                    suggestions.append("Add more challenging scenarios during RL training")
                elif fatigue_counts["Low"] == 0:
                    suggestions.append("Add more conservative scenarios during RL training")
        except Exception as e:
            logging.warning(f"Error analyzing session data for suggestions: {e}")
        
        # Add general suggestions if we have nothing specific
        if not suggestions:
            suggestions.append("Collect more varied workout sessions for better RL training data")
            suggestions.append("Increase training episodes for more robust learning")
        
        # Return strategy with guaranteed fields
        return {
            "suggestions": suggestions,
            "recommended_weights": {
                "safety_weight": min(1.0, self.safety_weight * 1.2) if safety_issues else self.safety_weight,
                "performance_weight": self.performance_weight * 0.9 if safety_issues else self.performance_weight,
                "adherence_weight": self.adherence_weight
            },
            "current_weights": {
                "safety_weight": self.safety_weight,
                "performance_weight": self.performance_weight,
                "adherence_weight": self.adherence_weight
            }
        }
    
    def start_session(self, user_profile=None):
        """Override start_session to initialize RL session data"""
        # Reset RL session data
        self.rl_recommendations = []
        self.traditional_recommendations = []
        self.current_set_num = 0
        
        # Call parent implementation
        super().start_session(user_profile)

    def _fix_model_input_shapes(self):
        """Fix input shape incompatibility issues between models"""
        # Check if the safety_enhanced model exists
        if hasattr(self, 'safety_enhanced_model') and self.safety_enhanced_model is not None:
            try:
                # Get the expected input shape from the model
                expected_shape = self.safety_enhanced_model.input_shape
                logging.info(f"Safety enhanced model expects input shape: {expected_shape}")
                
                # Add padding if needed during prediction
                self.safety_model_needs_padding = True
                self.safety_model_expected_shape = expected_shape
            except Exception as e:
                logging.warning(f"Could not determine safety model input shape: {e}")
                self.safety_model_needs_padding = False

    def _pad_input_for_safety_model(self, features):
        """Pad input data to match expected model shape"""
        if not hasattr(self, 'safety_model_needs_padding') or not self.safety_model_needs_padding:
            return features
        
        try:
            current_shape = features.shape
            logging.info(f"Current input shape: {current_shape}")
            
            expected_shape = self.safety_model_expected_shape[1:]  # Remove batch dimension
            
            # If we need more time steps (rows)
            if current_shape[1] < expected_shape[1]:
                padding_rows = expected_shape[1] - current_shape[1]
                # Pad with zeros
                padding = np.zeros((current_shape[0], padding_rows, current_shape[2]))
                padded_features = np.concatenate([features, padding], axis=1)
                logging.info(f"Padded rows from {current_shape} to {padded_features.shape}")
                
            # If we need more features (columns)
            if current_shape[2] < expected_shape[2]:
                padding_cols = expected_shape[2] - current_shape[2]
                # Create padding with zeros
                if 'padded_features' in locals():
                    padding = np.zeros((padded_features.shape[0], padded_features.shape[1], padding_cols))
                    padded_features = np.concatenate([padded_features, padding], axis=2)
                else:
                    padding = np.zeros((current_shape[0], current_shape[1], padding_cols))
                    padded_features = np.concatenate([features, padding], axis=2)
                logging.info(f"Padded columns to {padded_features.shape}")
                
            return padded_features if 'padded_features' in locals() else features
            
        except Exception as e:
            logging.error(f"Error padding input: {e}")
            return features

    def _format_rl_recommendation(self, rl_rec):
        """Format RL recommendation in a consistent way that can be parsed correctly"""
        if not rl_rec:
            return None
        
        # Ensure weight, reps, and rest_time are properly rounded numbers
        weight = float(round(rl_rec['weight'], 1))
        reps = int(rl_rec['reps'])
        rest_time = int(rl_rec['rest_time'])
        
        # Format with clear separators to avoid parsing issues
        formatted_rec = f"Next set: {weight} kg, {reps} reps, rest for {rest_time} seconds"
        
        return formatted_rec

    def get_safe_recommendation(self, user_profile, fatigue_level):
        """
        Get a safe recommendation using RL with fallback to traditional method
        
        Args:
            user_profile: User profile dictionary
            fatigue_level: Current fatigue level
            
        Returns:
            Recommendation string
        """
        try:
            # Try RL recommendation first
            if self.use_rl and self.rl_recommender is not None:
                try:
                    rec = self.get_recommendation(user_profile)
                    if rec:
                        self.debug_info['successful_rl_recommendations'] += 1
                        return rec
                except Exception as e:
                    logging.error(f"Error getting RL recommendation: {e}")
            
            # Fallback to traditional recommendation
            self.debug_info['fallback_recommendations'] += 1
            return self._generate_traditional_recommendation(user_profile, fatigue_level)
            
        except Exception as e:
            logging.error(f"Fatal error in RL recommendation: {e}")
            # Last resort fallback
            return self._generate_traditional_recommendation(user_profile, fatigue_level)

    def _generate_traditional_recommendation(self, user_profile, fatigue_level):
        """
        Generate a traditional recommendation based on user profile and fatigue level
        
        Args:
            user_profile: User profile dictionary
            fatigue_level: Current fatigue level
            
        Returns:
            Recommendation string with weight, reps, and rest time
        """
        try:
            # Get current values
            current_weight = float(user_profile.get('current_weight', 0.0))
            current_reps = int(user_profile.get('current_reps', 0))
            fitness_level = user_profile.get('fitness_level', 'Beginner')
            recovery_rate = float(user_profile.get('recovery_rate', 1.0))
            workout_goal = user_profile.get('workout_goal', 'Strength')
            
            # Base adjustments on fatigue level
            if fatigue_level == 'High':
                # Reduce weight and reps for high fatigue
                weight_adjustment = 0.85  # 15% reduction
                rep_adjustment = 0.8     # 20% reduction
                rest_time = 120          # Longer rest
            elif fatigue_level == 'Moderate':
                # Moderate adjustments
                weight_adjustment = 0.95  # 5% reduction
                rep_adjustment = 0.9     # 10% reduction
                rest_time = 90           # Medium rest
            else:  # Low fatigue
                # Small adjustments or increase
                weight_adjustment = 1.05  # 5% increase
                rep_adjustment = 1.0     # No change
                rest_time = 60           # Standard rest
            
            # Adjust for fitness level
            if fitness_level == 'Beginner':
                weight_adjustment *= 0.9  # More conservative
                rep_adjustment *= 0.9
            elif fitness_level == 'Advanced':
                weight_adjustment *= 1.1  # More aggressive
                rep_adjustment *= 1.1
            
            # Adjust for workout goal
            if workout_goal == 'Endurance':
                weight_adjustment *= 0.9
                rep_adjustment *= 1.2
                rest_time = int(rest_time * 0.8)  # Shorter rest
            elif workout_goal == 'Strength':
                weight_adjustment *= 1.1
                rep_adjustment *= 0.9
                rest_time = int(rest_time * 1.2)  # Longer rest
            
            # Apply recovery rate
            weight_adjustment *= recovery_rate
            rep_adjustment *= recovery_rate
            rest_time = int(rest_time / recovery_rate)
            
            # Calculate new values
            new_weight = round(current_weight * weight_adjustment, 1)
            new_reps = max(1, min(15, round(current_reps * rep_adjustment)))
            new_rest = max(30, min(180, rest_time))  # Keep rest time reasonable
            
            # Generate recommendation
            recommendation = f"Next set: {new_weight} kg, {new_reps} reps, rest for {new_rest} seconds"
            
            # Add insight based on velocity loss if available
            velocity_loss = user_profile.get('velocity_loss', None)
            if velocity_loss is not None:
                if velocity_loss < 0.1:
                    recommendation += "\nInsight: Low velocity loss indicates good endurance"
                elif velocity_loss > 0.2:
                    recommendation += "\nInsight: High velocity loss suggests fatigue"
            
            return recommendation
            
        except Exception as e:
            logging.error(f"Error generating traditional recommendation: {e}")
            # Return a very conservative recommendation as last resort
            return "Next set: 20.0 kg, 8 reps, rest for 90 seconds\nInsight: Using conservative fallback recommendation"