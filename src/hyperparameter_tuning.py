#src/hyperparameter_tuning.py
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import time
import matplotlib.pyplot as plt

def build_lstm_model(params, input_shape, num_classes=3):
    """
    Build LSTM model with given hyperparameters
    
    Args:
        params: Dictionary of hyperparameters
        input_shape: Shape of input data (timesteps, features)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Extract parameters
    lstm_units = params['lstm_units']
    dense_units = params['dense_units']
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Normalization
    x = BatchNormalization()(inputs)
    
    # LSTM layers
    x = LSTM(lstm_units[0], return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    
    if len(lstm_units) > 1:
        x = LSTM(lstm_units[1], return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)
    
    # Dense layers
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_hyperparameter_search(X, y, param_grid, n_folds=3, output_dir="results/hyperparameter_tuning"):
    """
    Run grid search for hyperparameter optimization
    
    Args:
        X: Input features
        y: Target labels
        param_grid: Dictionary of parameter values to try
        n_folds: Number of cross-validation folds
        output_dir: Directory to save results
        
    Returns:
        Dictionary with best parameters and results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize results storage
    all_results = []
    
    # Generate all parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    param_combinations = list(product(*param_values))
    print(f"Testing {len(param_combinations)} parameter combinations with {n_folds}-fold CV")
    
    # Set up cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Iterate through parameter combinations
    for i, combination in enumerate(param_combinations):
        # Create parameter dictionary
        params = {param_keys[j]: combination[j] for j in range(len(param_keys))}
        print(f"\nCombination {i+1}/{len(param_combinations)}: {params}")
        
        # Store fold results
        fold_accuracies = []
        fold_losses = []
        fold_times = []
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{n_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and train model
            model = build_lstm_model(params, input_shape=(X.shape[1], X.shape[2]))
            
            # Time training
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                epochs=20,  # Use fewer epochs for faster search
                batch_size=params.get('batch_size', 16),
                validation_data=(X_val, y_val),
                verbose=0
            )
            training_time = time.time() - start_time
            
            # Evaluate on validation set
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
            # Store results
            fold_accuracies.append(val_acc)
            fold_losses.append(val_loss)
            fold_times.append(training_time)
        
        # Calculate mean and std of metrics
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_loss = np.mean(fold_losses)
        mean_time = np.mean(fold_times)
        
        print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  Mean loss: {mean_loss:.4f}")
        print(f"  Mean training time: {mean_time:.2f}s")
        
        # Store results
        result = {
            'params': params,
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'mean_loss': float(mean_loss),
            'mean_training_time': float(mean_time)
        }
        all_results.append(result)
    
    # Find best parameters
    best_idx = np.argmax([r['mean_accuracy'] for r in all_results])
    best_result = all_results[best_idx]
    
    print("\nBest parameters:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")
    print(f"Mean accuracy: {best_result['mean_accuracy']:.4f}")
    
    # Save all results
    with open(output_path / "hyperparameter_search_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Plot results - more robust approach
    plt.figure(figsize=(12, 8))
    
    # Just plot the accuracy of all trials
    plt.subplot(2, 1, 1)
    trial_numbers = range(1, len(all_results) + 1)
    accuracies = [r['mean_accuracy'] for r in all_results]
    plt.plot(trial_numbers, accuracies, 'o-')
    plt.axhline(y=best_result['mean_accuracy'], color='r', linestyle='--', 
                label=f"Best: {best_result['mean_accuracy']:.4f}")
    plt.title('Accuracy vs Trial Number')
    plt.xlabel('Trial Number')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    
    # Plot parameters that are not lists - typically floats or integers
    plt.subplot(2, 1, 2)
    non_list_params = [param for param in param_grid.keys() 
                      if not isinstance(param_grid[param][0], list)]
    
    # If we have non-list parameters, plot them
    if non_list_params:
        # Pick the first non-list parameter to plot
        param = non_list_params[0]
        param_values = [r['params'][param] for r in all_results]
        
        # Group by parameter value
        unique_values = sorted(set(param_values))
        mean_accs = []
        
        for value in unique_values:
            indices = [i for i, v in enumerate(param_values) if v == value]
            mean_acc = np.mean([accuracies[i] for i in indices])
            mean_accs.append(mean_acc)
        
        plt.plot(unique_values, mean_accs, 'o-')
        plt.title(f'Accuracy vs {param}')
        plt.xlabel(param)
        plt.ylabel('Mean Accuracy')
    else:
        plt.text(0.5, 0.5, "No scalar parameters to plot", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path / "hyperparameter_search.png")
    
    return best_result