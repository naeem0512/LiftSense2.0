import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import Dict, Any, List, Optional
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.real_time_sliding_window import SlidingWindowPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoWorkoutSimulator:
    def __init__(self, 
                 model_path: str,
                 scaler_path: str,
                 data_path: str,
                 window_size: int = 5,
                 min_confidence_threshold: float = 0.4):
        """Initialize the demo workout simulator."""
        self.predictor = SlidingWindowPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            window_size=window_size,
            min_confidence_threshold=min_confidence_threshold
        )
        self.data_path = data_path
        self.results = {
            'predictions': [],
            'actual': [],
            'metrics': [],
            'accuracy': [],
            'confidence': []
        }
        
    def simulate_workout(self, num_reps: int = 30) -> Dict[str, Any]:
        """Simulate a workout session using augmented data."""
        # Load augmented data
        df = pd.read_csv(self.data_path)
        
        # Select a random user's workout session
        user_id = np.random.choice(df['User'].unique())
        workout_data = df[df['User'] == user_id].sort_values(['Set', 'Rep']).head(num_reps)
        
        # Process each rep
        for i, (_, row) in enumerate(workout_data.iterrows(), 1):
            logger.info(f"\nRep {i}/{num_reps}")
            
            # Get set and rep numbers
            set_num = row['Set']
            rep_num = row['Rep']
            logger.info(f"Set {set_num}, Rep {rep_num}")
            
            # Prepare rep data with only the features used in training
            rep_data = {
                'rep_speed': float(row['RepSpeed']),
                'force_output': float(row['ForceOutput']),
                'heart_rate': float(row['HeartRate']),
                'velocity_loss': float(row['VelocityLoss']),
                'recovery_rate': float(row['RecoveryRate']),
                'work_capacity': float(row['WorkCapacity']),
                'relative_intensity': float(row['RelativeIntensity']),
                'density': float(row['Density']),
                'grip_strength': float(row['GripStrength']),
                'fatigue_level': row['FatigueLevel']
            }
            
            # Get prediction
            predicted_fatigue, confidence = self.predictor.update_prediction_window(rep_data)
            actual_fatigue = rep_data['fatigue_level']
            
            logger.info(f"Predicted fatigue: {predicted_fatigue} (confidence: {confidence:.2f})")
            logger.info(f"Actual fatigue: {actual_fatigue}")
            
            # Log current metrics
            logger.info("Current metrics:")
            for metric, value in rep_data.items():
                if metric != 'fatigue_level':
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.2f}")
                    else:
                        logger.info(f"  {metric}: {value}")
            
            # Store results
            self.results['predictions'].append(predicted_fatigue)
            self.results['actual'].append(actual_fatigue)
            self.results['metrics'].append(rep_data)
            self.results['confidence'].append(confidence)
            
            # Calculate accuracy
            if predicted_fatigue != "Unknown":
                accuracy = 1.0 if predicted_fatigue == actual_fatigue else 0.0
            else:
                accuracy = 0.0
            self.results['accuracy'].append(accuracy)
        
        return self._get_summary()
    
    def _get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from the simulation results."""
        # Calculate prediction distribution
        pred_dist = pd.Series(self.results['predictions']).value_counts()
        
        # Calculate accuracy metrics
        valid_predictions = [p for p in self.results['predictions'] if p != "Unknown"]
        if valid_predictions:
            accuracy = sum(1 for p, a in zip(self.results['predictions'], self.results['actual']) 
                         if p != "Unknown" and p == a) / len(valid_predictions)
        else:
            accuracy = 0.0
            
        avg_confidence = np.mean([c for c in self.results['confidence'] if c > 0])
        
        return {
            'total_reps': len(self.results['predictions']),
            'final_accuracy': accuracy * 100,
            'average_confidence': avg_confidence * 100,
            'prediction_distribution': pred_dist.to_dict(),
            'results': self.results
        }
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Visualize the simulation results."""
        if not self.results['predictions']:
            logger.warning("No prediction history to visualize")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot accuracy over time
        ax1.plot(self.results['accuracy'], label='Accuracy', color='blue')
        ax1.set_title('Prediction Accuracy Over Time')
        ax1.set_xlabel('Rep Number')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True)
        ax1.legend()
        
        # Plot confidence over time
        ax2.plot(self.results['confidence'], label='Confidence', color='green')
        ax2.set_title('Prediction Confidence Over Time')
        ax2.set_xlabel('Rep Number')
        ax2.set_ylabel('Confidence')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def run_demo(model_path: str, 
            scaler_path: str,
            data_path: str,
            num_reps: int = 30,
            window_size: int = 5,
            min_confidence_threshold: float = 0.4,
            save_plot: bool = True) -> None:
    """Run the demo workout simulation."""
    try:
        # Initialize simulator
        simulator = DemoWorkoutSimulator(
            model_path=model_path,
            scaler_path=scaler_path,
            data_path=data_path,
            window_size=window_size,
            min_confidence_threshold=min_confidence_threshold
        )
        
        # Run simulation
        logger.info("Starting workout simulation...")
        summary = simulator.simulate_workout(num_reps)
        
        # Print summary
        logger.info("\nDemo Summary:")
        logger.info(f"Total reps: {summary['total_reps']}")
        logger.info(f"Final accuracy: {summary['final_accuracy']:.2f}%")
        logger.info(f"Average confidence: {summary['average_confidence']:.2f}%")
        
        logger.info("\nPrediction Distribution:")
        for pred, count in summary['prediction_distribution'].items():
            logger.info(f"{pred}: {count}")
        
        # Visualize results
        if save_plot:
            plot_path = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            simulator.visualize_results(save_path=plot_path)
            logger.info(f"\nResults plot saved to: {plot_path}")
        else:
            simulator.visualize_results()
            
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
        raise

def main():
    """Main entry point for the demo."""
    model_path = "models/new_fatigue_model.h5"
    scaler_path = "models/new_augmented_scaler.joblib"
    data_path = "data/augmented_workout_data.csv"
    
    run_demo(
        model_path=model_path,
        scaler_path=scaler_path,
        data_path=data_path,
        num_reps=30,
        window_size=5,
        min_confidence_threshold=0.4,
        save_plot=True
    )

if __name__ == "__main__":
    main() 