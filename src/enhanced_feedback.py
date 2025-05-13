#src/enhanced_feedback.py

from src.real_time_feedback import RealTimeFeedbackSystem
from src.ensemble_model_manager import EnsembleModelManager
import numpy as np
import logging
import re

class EnhancedFeedbackSystem(RealTimeFeedbackSystem):
    """
    Enhanced feedback system using ensemble prediction with safety-focused models
    """
    
    def __init__(
        self,
        ensemble_path="data/models/v2.0/ensemble",
        use_safety_models=True,  # New parameter
        **kwargs
    ):
        """Initialize with both base and ensemble capabilities
        
        Args:
            ensemble_path (str): Path to ensemble model directory
            use_safety_models (bool): Whether to use safety-enhanced model weights
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize base system
        super().__init__(**kwargs)
        
        # Add ensemble capability
        try:
            self.ensemble = EnsembleModelManager(ensemble_path)
            
            # Configure safety enhancements
            if use_safety_models:
                # Modify weights to emphasize safety-focused models
                self.ensemble.weights = {
                    'early_phase': 0.2,
                    'middle_phase': 0.2,
                    'late_phase': 0.2,
                    'early_phase_safety_model': 0.15,
                    'middle_phase_safety_model': 0.15,
                    'late_phase_safety_model': 0.1
                }
                logging.info("Using safety-enhanced model weights")
                
            self.use_ensemble = True
            logging.info("Ensemble model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load ensemble model: {e}")
            self.use_ensemble = False
    
    def _predict_fatigue(self, X):
        """Predict fatigue level using ensemble model with improved shape handling.
        
        Args:
            X: Input data of shape (batch_size, timesteps, features) or list of rep dicts
        Returns:
            Tuple of (fatigue_level, confidence)
        """
        # If X is a list of rep dicts, convert to numpy array
        if isinstance(X, list):
            if len(X) == 0:
                logging.error("Empty rep data list passed to _predict_fatigue.")
                return "Moderate", 0.5
            # Define the feature order (must match training pipeline)
            feature_keys = [
                "RepSpeed", "ForceOutput", "HeartRate", "RPE"
            ]
            # Optionally add more features if used in training
            # For now, pad to 12 features
            arr = np.zeros((1, 10, 12), dtype=np.float32)
            for i, rep in enumerate(X[:10]):
                for j, key in enumerate(feature_keys):
                    arr[0, i, j] = float(rep.get(key, 0))
            X = arr
            logging.info(f"Converted rep list to array: {X.shape}")
        
        try:
            # Log input shape for debugging
            logging.info(f"Input shape before processing: {X.shape}")
            
            # Ensure minimum timesteps
            if X.shape[1] < 3:
                X = np.pad(X, ((0, 0), (0, 3 - X.shape[1]), (0, 0)), mode='constant')
                logging.info(f"Padded input to minimum timesteps: {X.shape}")
            
            # Get ensemble predictions
            try:
                predictions, confidence_scores, uncertainty = self.ensemble.predict(X)
                
                # Convert predictions to class labels
                pred_classes = np.argmax(predictions, axis=1)
                fatigue_levels = [self.le_fatigue.inverse_transform([idx])[0] for idx in pred_classes]
                
                # Store confidence for RL
                self.last_confidence = float(confidence_scores[0])
                
                return fatigue_levels[0], self.last_confidence
                
            except Exception as e:
                logging.error(f"Ensemble prediction error: {str(e)}")
                logging.info("Falling back to standard model")
                
                # Fallback to standard model
                predictions = self.standard_model.predict(X)
                pred_class = np.argmax(predictions[0])
                fatigue_level = self.le_fatigue.inverse_transform([pred_class])[0]
                confidence = float(predictions[0][pred_class])
                
                # Store confidence for RL
                self.last_confidence = confidence
                
                return fatigue_level, confidence
                
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            # Return moderate fatigue as fallback
            return "Moderate", 0.5
            
    def analyze_set_data(self, set_data):
        """
        Analyze set data for insights beyond fatigue prediction
        
        Args:
            set_data: List of rep data dictionaries
            
        Returns:
            Dictionary of analysis results
        """
        if not set_data or len(set_data) < 2:
            return {"error": "Not enough reps to analyze"}
            
        # Extract key metrics
        speeds = [rep["RepSpeed"] for rep in set_data]
        forces = [rep["ForceOutput"] for rep in set_data]
        
        # Calculate velocity loss
        velocity_loss = ((speeds[0] - speeds[-1]) / speeds[0]) * 100 if speeds[0] > 0 else 0
        
        # Calculate force decay
        force_decay = ((forces[0] - forces[-1]) / forces[0]) * 100 if forces[0] > 0 else 0
        
        # Calculate mechanical work
        work = [s * f for s, f in zip(speeds, forces)]
        total_work = sum(work)
        
        # Calculate power trend
        power_trend = -1 if work[0] > work[-1] else 1
        
        # Early vs late rep comparison
        early_reps = set_data[:len(set_data)//3]
        late_reps = set_data[-len(set_data)//3:]
        
        if not early_reps or not late_reps:
            return {"error": "Not enough reps for early/late comparison"}
            
        early_avg_speed = sum([r["RepSpeed"] for r in early_reps]) / len(early_reps)
        late_avg_speed = sum([r["RepSpeed"] for r in late_reps]) / len(late_reps)
        
        # Generate insights
        insights = []
        
        if velocity_loss > 30:
            insights.append("Significant velocity loss indicates high fatigue")
        elif velocity_loss > 20:
            insights.append("Moderate velocity loss indicates some fatigue")
        else:
            insights.append("Low velocity loss indicates good endurance")
            
        if late_avg_speed / early_avg_speed < 0.7:
            insights.append("Substantial drop in speed during later reps")
        
        # Return analysis
        return {
            "velocity_loss": velocity_loss,
            "force_decay": force_decay,
            "total_work": total_work,
            "power_trend": power_trend,
            "early_vs_late_speed_ratio": late_avg_speed / early_avg_speed,
            "insights": insights
        }
    
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
            match = re.search(r"Next set: (\d+\.?\d*)kg, (\d+) reps", recommendation_text)
            if not match:
                return None
                
            weight = float(match.group(1))
            reps = int(match.group(2))
            
            # Look for rest time
            rest_match = re.search(r"rest for (\d+) seconds", recommendation_text)
            rest_time = int(rest_match.group(1)) if rest_match else 60
            
            return (weight, reps, rest_time)
        except (ValueError, IndexError, TypeError) as e:
            logging.warning(f"Error parsing recommendation: {e}")
            return None
            
    def _generate_recommendation(self, fatigue_level, user_profile):
        """
        Override the parent method to add enhanced recommendations
        
        Args:
            fatigue_level: Predicted fatigue level
            user_profile: User profile data
            
        Returns:
            Recommendation string
        """
        # First get the basic recommendation from parent
        basic_rec = super()._generate_recommendation(fatigue_level, user_profile)
        
        if not hasattr(self, 'current_set') or not self.current_set:
            return basic_rec
            
        # We have set data, use it for enhanced recommendation
        try:
            # Analyze the current set
            analysis = self.analyze_set_data(self.current_set)
            
            # Parse the basic recommendation
            parsed = self._parse_recommendation(basic_rec)
            if not parsed:
                return basic_rec
                
            weight, reps, rest_time = parsed
            
            # Refine recommendation based on analysis
            velocity_loss = analysis.get("velocity_loss", 0)
            early_late_ratio = analysis.get("early_vs_late_speed_ratio", 1.0)
            
            # Adjust weight more precisely based on velocity loss
            if velocity_loss > 35:
                weight *= 0.85  # Significant reduction
            elif velocity_loss > 25:
                weight *= 0.9   # Moderate reduction
            elif velocity_loss < 10:
                weight *= 1.05  # Slight increase
                
            # Adjust reps based on speed ratio
            if early_late_ratio < 0.6:
                reps = max(5, reps - 2)  # Reduce reps if substantial slowdown
            elif early_late_ratio > 0.9:
                reps = min(12, reps + 1)  # Increase reps if maintaining speed well
                
            # Format enhanced recommendation
            enhanced_rec = f"Next set: {weight:.1f} kg, {reps} reps, rest for {rest_time} seconds"
            
            # Add insight if available
            if analysis.get("insights"):
                enhanced_rec += f"\nInsight: {analysis['insights'][0]}"
                
            return enhanced_rec
                
        except Exception as e:
            logging.warning(f"Error generating enhanced recommendation: {e}")
            return basic_rec
            
    def _execute_set(self, user_profile: dict) -> None:
        """
        Execute a set with enhanced recommendation processing
        
        This overrides the parent method to handle recommendations correctly
        """
        try:
            # Call the parent implementation
            result = super()._execute_set(user_profile)
            
            # Get the recommendation string
            if result and 'recommendation' in result:
                # Parse the recommendation to extract weight
                parsed = self._parse_recommendation(result['recommendation'])
                if parsed:
                    weight, reps, _ = parsed
                    # Update user profile safely with the numeric weight
                    user_profile['current_weight'] = weight
                    user_profile['current_reps'] = reps
            
            return result
            
        except Exception as e:
            logging.error(f"Error in enhanced set execution: {e}")
            return None
        
    def start_session(self, user_profile=None):
        """Enhanced session with more detailed analytics"""
        # Use the parent method to run the session
        super().start_session(user_profile)
        
        # After session completes, we could add enhanced analytics
        # but that would require modifying the parent class more extensively
        # This can be expanded in future versions