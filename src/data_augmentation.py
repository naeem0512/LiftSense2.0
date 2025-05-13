import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_augmented_workout_data(
    num_users: int = 50,
    num_sets: int = 5,
    num_reps: int = 10,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate augmented workout data with enhanced variety and realism.
    
    Args:
        num_users: Number of unique users to generate
        num_sets: Number of sets per workout
        num_reps: Number of reps per set
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with augmented workout data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate total samples
    total_samples = num_users * num_sets * num_reps
    
    # Initialize data storage
    data = []
    fatigue_levels = {"Low": 0, "Moderate": 0, "High": 0}
    
    # Enhanced user characteristics
    fitness_levels = ["Beginner", "Intermediate", "Advanced"]
    fitness_probs = [0.3, 0.5, 0.2]  # More realistic distribution
    workout_goals = ["Strength", "Endurance", "Hypertrophy"]
    goal_probs = [0.4, 0.3, 0.3]
    
    for user in range(num_users):
        # Generate diverse user profiles
        fitness_level = np.random.choice(fitness_levels, p=fitness_probs)
        recovery_rate = np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3)
        workout_goal = np.random.choice(workout_goals, p=goal_probs)
        age = int(np.clip(np.random.normal(30, 8), 18, 50))
        gender = np.random.choice(["Male", "Female"], p=[0.6, 0.4])
        
        # Base capabilities with enhanced physiological limits
        if fitness_level == "Beginner":
            base_force = np.clip(np.random.normal(80, 15), 50, 120)
            base_speed = np.clip(np.random.normal(0.9, 0.15), 0.5, 1.3)
            hr_base = np.clip(np.random.normal(75, 5), 65, 85)
        elif fitness_level == "Intermediate":
            base_force = np.clip(np.random.normal(120, 20), 80, 180)
            base_speed = np.clip(np.random.normal(1.1, 0.15), 0.7, 1.5)
            hr_base = np.clip(np.random.normal(70, 5), 60, 80)
        else:  # Advanced
            base_force = np.clip(np.random.normal(160, 25), 120, 220)
            base_speed = np.clip(np.random.normal(1.3, 0.15), 0.9, 1.7)
            hr_base = np.clip(np.random.normal(65, 5), 55, 75)
        
        # Gender adjustments with more realistic factors
        if gender == "Female":
            base_force *= 0.75  # More realistic strength difference
            base_speed *= 1.05
            hr_base += 5  # Slightly higher base HR
        
        # Workout-specific adjustments
        if workout_goal == "Endurance":
            base_force *= 0.8
            base_speed *= 1.1
            hr_base += 5
        elif workout_goal == "Strength":
            base_force *= 1.1
            base_speed *= 0.9
            hr_base -= 3
        
        # Generate workout data
        for set_num in range(num_sets):
            # Set-specific fatigue progression
            set_fatigue = set_num * 0.15
            set_hr_increase = set_num * 8
            
            # Add set-specific variations
            set_variation = np.random.normal(0, 0.05)  # Random variation per set
            
            for rep_num in range(num_reps):
                # Enhanced fatigue modeling
                rep_fatigue = set_fatigue + (rep_num * 0.04)
                fatigue_factor = np.clip(
                    rep_fatigue + 
                    np.random.normal(0, 0.03) +  # More variation
                    set_variation,  # Set-specific variation
                    0, 0.85
                )
                
                # Improved force-velocity relationship
                speed = np.clip(
                    base_speed * (1 - fatigue_factor**1.3) * 
                    (1 + np.random.normal(0, 0.02)),  # Small random variation
                    0.2, 2.0
                )
                
                force = np.clip(
                    base_force * 
                    (1 - fatigue_factor**0.8) * 
                    recovery_rate * 
                    (1 + np.random.normal(0, 0.03)),  # More variation in force
                    20, 250
                )
                
                # Enforce inverse relationship with more realistic curve
                if speed < 0.5:
                    force = min(force, speed * 250)  # Adjusted force-velocity curve
                
                # Enhanced HR response
                hr = np.clip(
                    hr_base + 
                    set_hr_increase + 
                    rep_num * 2 + 
                    np.random.randint(-4, 5) +  # More variation
                    (8 if workout_goal == "Endurance" else 0),  # Goal-specific adjustment
                    60, 208 - 0.7 * age - 5  # Age-adjusted max HR
                )
                
                # Improved grip strength calculation
                grip = np.clip(
                    (50 - (set_num * 8) - (rep_num * 1.2)) * 
                    (1.1 if gender == "Male" else 1.0) * 
                    (1 + np.random.normal(0, 0.05)),  # Added variation
                    15, 50
                )
                
                # Calculate velocity loss
                if rep_num == 0:
                    set_base_speed = speed
                velocity_loss = (set_base_speed - speed) / set_base_speed * 100
                
                # Enhanced fatigue level assignment
                if velocity_loss < 15:
                    fatigue_level = "Low"
                elif velocity_loss < 30:
                    fatigue_level = "Moderate"
                else:
                    fatigue_level = "High"
                fatigue_levels[fatigue_level] += 1
                
                # Add workout-specific metrics
                work_capacity = speed * force
                relative_intensity = force / (grip + 1e-6)
                density = work_capacity / (recovery_rate + 1e-6)
                
                data.append([
                    user, fitness_level, recovery_rate, workout_goal, age, gender,
                    set_num, rep_num, speed, force, hr, grip, fatigue_level,
                    velocity_loss, work_capacity, relative_intensity, density
                ])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        "User", "FitnessLevel", "RecoveryRate", "WorkoutGoal", "Age", "Gender",
        "Set", "Rep", "RepSpeed", "ForceOutput", "HeartRate", "GripStrength",
        "FatigueLevel", "VelocityLoss", "WorkCapacity", "RelativeIntensity",
        "Density"
    ])
    
    # Balance fatigue levels if needed
    df = balance_fatigue_levels(df)
    
    # Verify and print statistics
    logger.info("\nFatigue Level Distribution:")
    logger.info(df['FatigueLevel'].value_counts())
    
    logger.info("\nData Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique users: {df['User'].nunique()}")
    logger.info(f"Speed range: {df['RepSpeed'].min():.2f} - {df['RepSpeed'].max():.2f}")
    logger.info(f"Force range: {df['ForceOutput'].min():.1f} - {df['ForceOutput'].max():.1f}")
    logger.info(f"HR range: {df['HeartRate'].min()} - {df['HeartRate'].max()}")
    
    return df

def balance_fatigue_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance fatigue levels in the dataset while preserving patterns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Balanced DataFrame
    """
    # Get current distribution
    fatigue_counts = df['FatigueLevel'].value_counts()
    target_count = int(len(df) / 3)  # Equal distribution
    
    # Find the level with the most samples
    max_level = fatigue_counts.idxmax()
    max_count = fatigue_counts[max_level]
    
    # If any level has more than 5% deviation from target, balance
    if max_count > target_count * 1.05:
        # Get indices for each level
        level_indices = {
            level: df[df['FatigueLevel'] == level].index 
            for level in ['Low', 'Moderate', 'High']
        }
        
        # Randomly sample to target count
        balanced_indices = []
        for level in ['Low', 'Moderate', 'High']:
            if len(level_indices[level]) > target_count:
                # Randomly select target_count samples
                selected = np.random.choice(
                    level_indices[level], 
                    size=target_count, 
                    replace=False
                )
            else:
                # Keep all samples if under target
                selected = level_indices[level]
            balanced_indices.extend(selected)
        
        # Create balanced dataset
        df = df.loc[balanced_indices].reset_index(drop=True)
        
        logger.info("\nBalanced Fatigue Distribution:")
        logger.info(df['FatigueLevel'].value_counts())
    
    return df

def save_augmented_data(
    df: pd.DataFrame,
    output_path: str = "data/augmented_workout_data.csv",
    save_scaler: bool = True
) -> None:
    """
    Save augmented data and optional scaler.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the data
        save_scaler: Whether to save a fitted scaler
    """
    # Create output directory if needed
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Save data
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved augmented data to {output_path}")
    
    # Save scaler if requested
    if save_scaler:
        # Select numeric columns for scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'User']
        
        # Fit and save scaler
        scaler = StandardScaler()
        scaler.fit(df[numeric_cols])
        
        scaler_path = str(Path(output_path).parent / "augmented_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    # Generate and save augmented data
    df = generate_augmented_workout_data(
        num_users=50,  # More users for better variety
        num_sets=5,
        num_reps=10,
        random_seed=42
    )
    
    save_augmented_data(df) 