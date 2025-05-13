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

The system consists of the following core components:

1. **Web Interface** (`app.py`)
   - Main user interface
   - Session management
   - Real-time feedback

2. **Enhanced Sensors** (`enhanced_sensors.py`)
   - Sensor configuration and management
   - Data collection and processing
   - Simulation capabilities

3. **Technique Analyzer** (`technique_analyzer.py`)
   - Movement pattern recognition
   - Form assessment
   - Feedback generation

4. **Advanced Models** (`advanced_models.py`)
   - Neural network architectures
   - Transfer learning
   - Confidence scoring

5. **Database Manager** (`database_manager.py`)
   - User and session storage
   - Progress tracking
   - Workout planning

6. **API Interface** (`api.py`)
   - RESTful endpoints
   - Mobile integration
   - External system connectivity

7. **Data Visualization** (`dashboard.py`)
   - Performance trends
   - Form analysis visualizations
   - Sensor data exploration

## Extending the System

### Adding New Exercises

To add a new exercise type:

1. Create a movement pattern definition in `technique_analyzer.py`:

```python
# Add to TechniqueAnalyzer.STANDARD_EXERCISES
'new_exercise': {
    'concentric': {
        'accel_ranges': {'x': (0.5, 1.5), 'y': (-0.5, 0.5), 'z': (-0.3, 0.3)},
        'gyro_ranges': {'x': (-50, 50), 'y': (-30, 30), 'z': (-20, 20)},
        'emg_channels': {'primary': [0, 1], 'secondary': [], 'stabilizers': [4, 5]}
    },
    'eccentric': {
        'accel_ranges': {'x': (0.2, 1.2), 'y': (-0.3, 0.3), 'z': (-0.2, 0.2)},
        'gyro_ranges': {'x': (-30, 30), 'y': (-20, 20), 'z': (-15, 15)},
        'emg_channels': {'primary': [0, 1], 'secondary': [], 'stabilizers': [4, 5]}
    }
}
```

2. Add the exercise to the UI in `app.py`:

```python
# Add to exercise_dropdown options
{"label": "New Exercise", "value": "new_exercise"}
```

### Creating Custom Models

To add a custom model architecture:

1. Create your model in `advanced_models.py`:

```python
def build_custom_model(self) -> tf.keras.Model:
    """Build a custom model architecture"""
    input_shape = self.config["input_shape"]
    num_classes = self.config["num_classes"]
    
    # Define your model architecture
    model = Sequential([
        # Your layers here
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=self.config["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

2. Add it to the ModelManager class:

```python
# Add to the __init__ method
self.custom_model = self.build_custom_model()
```

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

1. **Services won't start**
   - Check if ports are in use: `lsof -i :5000`
   - Ensure data directories exist: `mkdir -p data/models`
   - Check logs: `cat logs/liftsense.log`

2. **No sensor data**
   - Check sensor connection
   - Try simulation mode: Edit `config.json` and set `"simulation_mode": true`
   - Check permissions: `chmod +x enhanced_sensors.py`

3. **Model prediction errors**
   - Ensure models are trained: `python -m tools.train_models`
   - Check model compatibility: Delete `data/models` folder and restart
   - Try base model: Delete user-specific model in `data/user_models`

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