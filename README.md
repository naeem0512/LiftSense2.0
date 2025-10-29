# LiftSense 2.0

An advanced workout analysis system with real-time form feedback, personalized recommendations, and comprehensive performance tracking.

## Features

### Enhanced User Experience
- Web-based dashboard with intuitive interface
- Real-time visualization of workout data
- Comprehensive exercise performance reports
- Mobile-friendly design

### Advanced Data Collection
- Support for multiple sensor types (EMG, IMU, heart rate)
- High-frequency data capture (up to 500Hz)
- Form/technique analysis using motion patterns
- Muscle activation assessment

### Machine Learning Enhancements
- Multiple model architectures (GRU, CNN-LSTM, Transformer)
- Personalized adaptation through transfer learning
- Confidence scoring for reliability assessment
- Ensemble prediction for improved accuracy

### Performance Tracking
- SQLite database for workout history
- Progress tracking across multiple metrics
- Personal records identification
- Technique issue tracking and resolution

### Integration Capabilities
- REST API for mobile app integration
- Data export/import functionality
- Compatibility with external fitness ecosystems

## System Requirements

- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge)
- 2GB RAM minimum (4GB recommended)
- 100MB disk space minimum
- Internet connection (optional, for updates)

## Installation

### Option 1: Using pip (Recommended)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from PyPI
pip install liftsense

# Initialize system
liftsense init
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/liftsense.git
cd liftsense

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize system
python main.py init
```

## Getting Started

### Starting the System

```bash
# Start all services
python main.py start

# Run a demo
python main.py demo

# Stop services
python main.py stop
```

### Accessing the Interfaces

- Web Dashboard: http://localhost:5000
- Data Visualization: http://localhost:5002
- API Documentation: http://localhost:5001/api/docs

## Using the System

### Web Dashboard

1. **Create a user account**
   - Click "Create User" on the welcome screen
   - Fill in your profile information
   - Click "Save"

2. **Start a workout session**
   - Select an exercise
   - Configure weight, reps, and sets
   - Click "Start Session"

3. **View workout results**
   - After completion, view your form score
   - Check technique feedback
   - Review recommendations for next workout

### API Integration

The system provides a RESTful API for integration with mobile apps and external systems.

Example API usage:

```python
import requests
import json

# Create a user
response = requests.post(
    "http://localhost:5001/api/users",
    json={
        "name": "John Doe",
        "age": 30,
        "gender": "Male",
        "fitness_level": "Intermediate"
    }
)

user_id = response.json()["user_id"]

# Create a session
response = requests.post(
    "http://localhost:5001/api/sessions",
    json={
        "user_id": user_id,
        "exercise": "bench_press",
        "timestamp": "2025-05-10T14:30:00",
        "rep_count": 10,
        "set_count": 3,
        "weight": 100.0
    }
)

# Get workout recommendations
response = requests.get(
    f"http://localhost:5001/api/recommendations/{user_id}/bench_press"
)

recommendations = response.json()
print(f"Recommended weight: {recommendations['weight']}kg")
print(f"Recommended reps: {recommendations['reps']}")
```

## Component Architecture

The repository now focuses on the streamlined research workflow that powers the fatigue model. The core pieces are:

1. **Main pipeline (`run.py`)**
   - Orchestrates data simulation, preprocessing, model training, and evaluation.
   - Boots the real-time feedback system with the freshly trained artefacts.

2. **Data preprocessing (`src/data_preprocessing.py`)**
   - Generates synthetic workout logs with strict physiological constraints.
   - Handles scaling, label encoding, and dataset splits used across the project.

3. **Model definition (`src/model.py`)**
   - Hosts the CNN-LSTM architecture and attention blocks used for fatigue prediction.
   - Provides a single entry point for tailoring layers and regularisation.

4. **Training utilities (`src/model_training.py`)**
   - Contains the training loop, callbacks, and augmentation helpers that power `run.py`.
   - Exposes `build_lstm_model` and `train_model` for reuse in notebooks or experiments.

5. **Real-time feedback (`src/real_time_feedback.py`)**
   - Streams inference results, interprets fatigue states, and generates actionable coaching cues.
   - Integrates with the bio-signal simulator for hardware-free experimentation.

6. **Bio-signal simulation (`src/biosignal_simulator.py`)**
   - Synthesises ECG, heart rate, acceleration, and EDA signals for end-to-end demos.

7. **Calibration tools (`src/model_calibration.py`)**
   - Implements temperature scaling and uncertainty estimation for safer deployment.

8. **Shared utilities (`src/utils.py`)**
   - Centralises logging configuration and expert heuristics for recommendation text.

## Extending the System

### Customising the model architecture

The default network lives in `src/model.py`. To experiment with a different layout, modify `create_model` or add a new helper alongside it:

```python
from src import model

def create_compact_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

# Use the custom model during training
custom_model = create_compact_model(input_shape=(12, 10), num_classes=3)
```

Pass the custom model into `train_model` from `src/model_training.py` to reuse the optimisation logic and callbacks.

### Adapting data generation

Workout synthesis happens in `simulate_workout_data` within `src/data_preprocessing.py`. You can tweak class balances, add new sensor channels, or pipe in recorded data before handing the arrays to `preprocess_data`.

## Configuration

The system can be configured through the `config/config.json` file. Key settings include:

```json
{
    "data_dir": "data",
    "model_dir": "data/models",
    "logs_dir": "logs",
    "web_port": 5000,
    "api_port": 5001,
    "enable_web": true,
    "enable_api": true,
    "enable_dashboard": true,
    "debug_mode": false,
    "simulation_mode": true
}
```

## Troubleshooting

### Common Issues

1. **`run.py` exits immediately**
   - Verify dependencies are installed: `pip install -r requirements.txt`
   - Remove corrupted artefacts and retry: `rm -rf data/models && python run.py`
   - Inspect execution logs at `logs/execution.log` for stack traces.

2. **Synthetic data generation fails**
   - Confirm there is disk space for the CSV export in `data/`.
   - Relax the force-velocity thresholds inside `simulate_workout_data` if you heavily customise the distributions.

3. **Real-time feedback shows stale outputs**
   - Delete cached scalers/encoders in `data/models` so that `run.py` rebuilds them.
   - When using hardware, fall back to the bundled simulator by enabling `simulation_mode` in `CONFIG` inside `run.py`.

### Getting Help

- Open an issue on GitHub
- Check the documentation
- Contact support at liftsense-support@example.com

## Contributing

Contributions are welcome! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Special thanks to the TensorFlow and Flask communities
- Inspired by research in biomechanics and exercise science