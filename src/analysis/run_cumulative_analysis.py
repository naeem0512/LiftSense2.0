import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Any
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.cumulative_fatigue import CumulativeFatigueAnalyzer
from src.data_preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert NumPy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def run_cumulative_analysis(
    data_path: str,
    output_dir: str,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run cumulative fatigue analysis on workout data.
    
    Args:
        data_path: Path to workout data CSV
        output_dir: Directory to save analysis results
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = pd.read_csv(data_path)
        
        # Initialize analyzer
        analyzer = CumulativeFatigueAnalyzer(window_sizes=[3, 5, 10])
        
        # Calculate cumulative features
        logger.info("Calculating cumulative features...")
        df_with_features = analyzer.calculate_cumulative_features(df)
        
        # Analyze set progression
        logger.info("Analyzing set progression...")
        progression_results = analyzer.analyze_set_progression(df_with_features)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        analyzer.visualize_cumulative_patterns(
            df_with_features,
            save_dir=str(output_path)
        )
        
        # Save numerical results
        results = {
            'progression_rates': convert_to_serializable(progression_results['progression_rates']),
            'feature_summary': {
                'n_users': int(df['User'].nunique()),
                'n_sets': int(df['Set'].max()),
                'n_reps': int(df['Rep'].max()),
                'fatigue_levels': df['FatigueLevel'].unique().tolist()
            }
        }
        
        # Convert results to JSON-serializable format
        results = convert_to_serializable(results)
        
        # Save results to JSON
        with open(output_path / 'cumulative_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save processed data
        df_with_features.to_csv(output_path / 'processed_data.csv', index=False)
        
        logger.info("\nAnalysis complete! Results saved to: %s", output_dir)
        logger.info("\nKey findings:")
        
        # Print key findings
        for metric, rates in progression_results['progression_rates'].items():
            if rates['p_value'] < 0.05:  # Statistically significant
                logger.info(f"\n{metric}:")
                logger.info(f"  Slope: {rates['slope']:.4f}")
                logger.info(f"  R-squared: {rates['r_squared']:.4f}")
                logger.info(f"  P-value: {rates['p_value']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during cumulative analysis: {str(e)}")
        raise

def main():
    # Set up paths
    data_path = project_root / 'data/workout_data.csv'
    output_dir = project_root / 'results/cumulative_analysis'
    
    # Run analysis
    run_cumulative_analysis(
        data_path=str(data_path),
        output_dir=str(output_dir)
    )

if __name__ == '__main__':
    main() 