import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CumulativeFatigueAnalyzer:
    def __init__(self, window_sizes: List[int] = [3, 5, 10]):
        """
        Initialize the cumulative fatigue analyzer.
        
        Args:
            window_sizes: List of window sizes for rolling calculations
        """
        self.window_sizes = window_sizes
        
    def calculate_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced cumulative fatigue features.
        
        Args:
            df: Input DataFrame with workout data
            
        Returns:
            DataFrame with additional cumulative features
        """
        df = df.copy()
        
        # Calculate work capacity as a function of force output and rep speed
        df['work_capacity'] = df['ForceOutput'] * df['RepSpeed']
        
        # Group by user and set to calculate set-level metrics
        df['set_id'] = df.groupby(['User', 'Set']).ngroup()
        
        # Basic cumulative metrics
        df['cumulative_work'] = df.groupby('User')['work_capacity'].cumsum()
        df['cumulative_reps'] = df.groupby('User')['Rep'].cumsum()
        df['cumulative_sets'] = df.groupby('User')['Set'].cumsum()
        
        # Set-level metrics
        set_metrics = df.groupby('set_id').agg({
            'RepSpeed': ['mean', 'std', 'min'],
            'ForceOutput': ['mean', 'std', 'min'],
            'HeartRate': ['mean', 'max'],
            'VelocityLoss': 'mean',
            'RecoveryRate': 'mean'
        }).reset_index()
        
        set_metrics.columns = ['set_id', 
                             'set_avg_speed', 'set_speed_std', 'set_min_speed',
                             'set_avg_force', 'set_force_std', 'set_min_force',
                             'set_avg_hr', 'set_max_hr',
                             'set_avg_vel_loss', 'set_avg_recovery']
        
        # Merge set metrics back to main dataframe
        df = df.merge(set_metrics, on='set_id', how='left')
        
        # Calculate rolling features for multiple window sizes
        for window in self.window_sizes:
            # Velocity-based metrics
            df[f'rolling_velocity_{window}'] = df.groupby('User')['RepSpeed'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'velocity_decay_{window}'] = df.groupby('User')['RepSpeed'].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
            
            # Force-based metrics
            df[f'rolling_force_{window}'] = df.groupby('User')['ForceOutput'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'force_decay_{window}'] = df.groupby('User')['ForceOutput'].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
            
            # Recovery metrics
            df[f'rolling_recovery_{window}'] = df.groupby('User')['RecoveryRate'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Fatigue accumulation
            df[f'fatigue_accumulation_{window}'] = df.groupby('User')['FatigueLevel'].transform(
                lambda x: x.rolling(window=window, min_periods=1).count()
            )
        
        # Calculate set-to-set fatigue progression
        df['set_fatigue_progression'] = df.groupby('User')['set_avg_vel_loss'].transform(
            lambda x: x.pct_change()
        )
        
        # Calculate recovery efficiency between sets
        df['inter_set_recovery'] = df.groupby('User')['set_avg_recovery'].transform(
            lambda x: x.diff()
        )
        
        return df
    
    def analyze_set_progression(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how fatigue patterns progress across sets.
        
        Args:
            df: DataFrame with cumulative features
            
        Returns:
            Dictionary with set progression analysis results
        """
        results = {}
        
        # Group by set number and fatigue level
        set_analysis = df.groupby(['Set', 'FatigueLevel']).agg({
            'set_avg_speed': 'mean',
            'set_avg_force': 'mean',
            'set_avg_vel_loss': 'mean',
            'set_avg_recovery': 'mean',
            'set_fatigue_progression': 'mean'
        }).reset_index()
        
        # Calculate fatigue progression rates
        progression_rates = {}
        for fatigue_level in df['FatigueLevel'].unique():
            level_data = set_analysis[set_analysis['FatigueLevel'] == fatigue_level]
            
            # Calculate progression rate for each metric
            for metric in ['set_avg_speed', 'set_avg_force', 'set_avg_vel_loss']:
                slope, _, r_value, p_value, _ = stats.linregress(
                    level_data['Set'], 
                    level_data[metric]
                )
                
                progression_rates[f'{metric}_fatigue_{fatigue_level}'] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
        
        results['progression_rates'] = progression_rates
        results['set_analysis'] = set_analysis
        
        return results
    
    def visualize_cumulative_patterns(self, df: pd.DataFrame, save_dir: Optional[str] = None):
        """
        Create visualizations of cumulative fatigue patterns.
        
        Args:
            df: DataFrame with cumulative features
            save_dir: Optional directory to save plots
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')  # Using a valid seaborn-style matplotlib theme
        
        # 1. Set progression plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Set', y='set_avg_vel_loss', hue='FatigueLevel')
        plt.title('Velocity Loss Progression Across Sets')
        plt.xlabel('Set Number')
        plt.ylabel('Average Velocity Loss')
        if save_dir:
            plt.savefig(f'{save_dir}/set_velocity_progression.png')
        plt.close()
        
        # 2. Rolling metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot rolling velocity for different window sizes
        for window in self.window_sizes:
            sns.lineplot(
                data=df, 
                x='Rep', 
                y=f'rolling_velocity_{window}',
                hue='FatigueLevel',
                ax=axes[0,0]
            )
        axes[0,0].set_title('Rolling Velocity Across Reps')
        axes[0,0].set_xlabel('Rep Number')
        axes[0,0].set_ylabel('Rolling Velocity')
        
        # Plot velocity decay
        for window in self.window_sizes:
            sns.lineplot(
                data=df,
                x='Rep',
                y=f'velocity_decay_{window}',
                hue='FatigueLevel',
                ax=axes[0,1]
            )
        axes[0,1].set_title('Velocity Decay Rate')
        axes[0,1].set_xlabel('Rep Number')
        axes[0,1].set_ylabel('Decay Rate')
        
        # Plot fatigue accumulation
        for window in self.window_sizes:
            sns.lineplot(
                data=df,
                x='Rep',
                y=f'fatigue_accumulation_{window}',
                hue='FatigueLevel',
                ax=axes[1,0]
            )
        axes[1,0].set_title('Fatigue Accumulation')
        axes[1,0].set_xlabel('Rep Number')
        axes[1,0].set_ylabel('Accumulated Fatigue')
        
        # Plot recovery efficiency
        sns.lineplot(
            data=df,
            x='Set',
            y='inter_set_recovery',
            hue='FatigueLevel',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Inter-Set Recovery')
        axes[1,1].set_xlabel('Set Number')
        axes[1,1].set_ylabel('Recovery Rate Change')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/cumulative_patterns.png')
        plt.close()
        
        # 3. Statistical significance plot
        plt.figure(figsize=(10, 6))
        set_analysis = df.groupby(['Set', 'FatigueLevel'])['set_avg_vel_loss'].mean().unstack()
        set_analysis.plot(marker='o')
        plt.title('Velocity Loss by Set and Fatigue Level')
        plt.xlabel('Set Number')
        plt.ylabel('Average Velocity Loss')
        plt.legend(title='Fatigue Level')
        if save_dir:
            plt.savefig(f'{save_dir}/fatigue_significance.png')
        plt.close() 