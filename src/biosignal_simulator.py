#src/biosignal_simulator.py

import numpy as np
from scipy.signal import resample
from biosppy.signals import tools

class BioSignalGenerator:
    """
    Medical-grade physiological signal generator with:
    - Realistic ECG waveforms
    - Clinically valid HR profiles
    - Synchronized movement patterns
    """

    def __init__(self, sampling_rate=100, hr_sampling_rate=1):
        self.sampling_rate = sampling_rate  # For ECG/EDA/Accel
        self.hr_sampling_rate = hr_sampling_rate  # For heart rate
        self.ecg_template = self._create_ecg_template()

    def _create_ecg_template(self):
        """Create realistic ECG template with artifacts"""
        t = np.linspace(0, 1, self.sampling_rate)  # Changed from ecg_sampling_rate to sampling_rate
        
        # Add baseline wander
        baseline = 0.1 * np.sin(2*np.pi*t/5)

        # Add more realistic noise components
        emg_noise = np.random.normal(0, 0.05, len(t))  # Muscle noise
        powerline_noise = 0.02 * np.sin(2*np.pi*50*t)  # 50Hz interference

        return (
            baseline + 
            emg_noise + 
            powerline_noise +
            0.25 * np.exp(-((t - 0.15)/0.05)**2) +  # P wave
            1.20 * np.exp(-((t - 0.35)/0.02)**2) +  # QRS complex
            0.30 * np.exp(-((t - 0.45)/0.10)**2)    # T wave
        )

    def generate_workout_session(self, duration_min=30, intensity=0.7):
        """Generates synchronized biometric data"""
        duration_sec = duration_min * 60
        return {
            "timestamps": np.linspace(0, duration_sec, 
                                     int(duration_sec * self.hr_sampling_rate)),
            "heart_rate": self._generate_hr_profile(duration_sec, intensity),
            "ecg": self._generate_ecg(duration_sec),
            "acceleration": self._generate_movement_pattern(duration_sec, intensity),
            "eda": self._simulate_eda(duration_sec, intensity)
        }

    def _generate_hr_profile(self, duration_sec, intensity):
        """Generates heart rate with physiological patterns"""
        t = np.linspace(0, duration_sec, int(duration_sec * self.hr_sampling_rate))
        
        # Physiological constraints
        age = 30  # Can parameterize
        max_hr = 208 - 0.7 * age  # Tanaka formula
        base_hr = np.clip(65 + (intensity * 15), 60, 80)
        peak_hr = np.clip(base_hr + (0.7 * (max_hr - base_hr)), base_hr+20, max_hr-10)

        # Phase transitions
        warmup = 0.5 * (1 + np.tanh((t - 0.1*duration_sec)/(duration_sec/10)))
        cooldown = 0.5 * (1 - np.tanh((t - 0.8*duration_sec)/(duration_sec/10)))

        # Exercise dynamics
        exercise = 0.7 + 0.3 * np.sin(2*np.pi*t/17)  # 17s cycles
        hr = base_hr + (peak_hr-base_hr) * (warmup * exercise - cooldown)
        
        # Add variability
        hr += 4 * np.sin(2*np.pi*t/5)  # Respiratory
        hr += np.random.normal(0, 1.5, len(t))  # Noise
        return np.clip(hr, 60, max_hr).astype(int)

    def _generate_ecg(self, duration_sec):
        """Generates continuous ECG synchronized with HR"""
        n_samples = int(duration_sec * self.sampling_rate)
        ecg_signal = np.zeros(n_samples)
        
        # Dynamic RR intervals based on current HR
        time_points = np.arange(0, duration_sec, 1/self.hr_sampling_rate)
        hr_inst = self._generate_hr_profile(duration_sec, 0.7)  # Reference profile
        rr_intervals = (60 / hr_inst) * self.sampling_rate  # Samples/beat
        
        # Place templates
        pos = 0
        while pos < n_samples:
            end = min(pos + len(self.ecg_template), n_samples)
            ecg_signal[pos:end] += self.ecg_template[:end-pos]
            pos += int(np.interp(pos/n_samples, time_points/duration_sec, rr_intervals))
            
        return {
            "filtered": ecg_signal,
            "rpeaks": np.where(ecg_signal > 0.8 * np.max(ecg_signal))[0]
        }

    def _generate_movement_pattern(self, duration_sec, intensity):
        """Simulates 3D accelerometer data"""
        t = np.linspace(0, duration_sec, int(duration_sec * self.sampling_rate))
        return np.column_stack([
            np.sin(2 * np.pi * (0.5 + intensity*2) * t) * (0.5 + intensity),  # X
            0.7 * np.cos(2 * np.pi * (0.5 + intensity*1.5) * t) * intensity,  # Y
            np.random.normal(0, 0.1, len(t))  # Z (noise)
        ])

    def _simulate_eda(self, duration_sec, intensity):
        """Simulates electrodermal activity"""
        n_samples = int(duration_sec * self.sampling_rate)
        eda = np.random.normal(0.5, 0.1, n_samples)
        
        # Add phasic responses
        for _ in range(int(8 * intensity)):
            onset = np.random.randint(0, n_samples - 100)
            eda[onset:onset+100] += 0.3 * np.exp(-np.linspace(0, 5, 100))
            
        return eda

    def generate_fatigue_progression(self, duration_min=45, sets=5):
        """Generate workout with progressive fatigue states"""
        set_duration = duration_min / sets
        full_session_data = {
            "timestamps": np.array([]),
            "heart_rate": np.array([]),
            "ecg": {"filtered": np.array([]), "rpeaks": np.array([])},
            "acceleration": None,  # Will be initialized as 2D array
            "eda": np.array([])
        }
        last_timestamp_offset = 0.0

        for i in range(sets):
            intensity = 0.5 + (i * 0.1)  # Intensity increases with each set
            set_data = self.generate_workout_session(
                duration_min=set_duration,
                intensity=intensity
            )
            
            current_timestamps = set_data["timestamps"] + last_timestamp_offset
            full_session_data["timestamps"] = np.append(full_session_data["timestamps"], current_timestamps)
            full_session_data["heart_rate"] = np.append(full_session_data["heart_rate"], set_data["heart_rate"])
            
            # Append ECG
            ecg_len_offset = len(full_session_data["ecg"]["filtered"])
            full_session_data["ecg"]["filtered"] = np.append(full_session_data["ecg"]["filtered"], set_data["ecg"]["filtered"])
            if len(set_data["ecg"]["rpeaks"]) > 0:  # Ensure rpeaks exist before adding offset
                full_session_data["ecg"]["rpeaks"] = np.append(full_session_data["ecg"]["rpeaks"], set_data["ecg"]["rpeaks"] + ecg_len_offset)
            
            # Append acceleration
            if full_session_data["acceleration"] is None:
                full_session_data["acceleration"] = set_data["acceleration"]
            else:
                full_session_data["acceleration"] = np.vstack([full_session_data["acceleration"], set_data["acceleration"]])

            # Append EDA
            full_session_data["eda"] = np.append(full_session_data["eda"], set_data["eda"])

            if len(current_timestamps) > 0:
                last_timestamp_offset = current_timestamps[-1]  # Update for next iteration
            else:  # if a set somehow generated no timestamps, at least advance by duration
                last_timestamp_offset += set_duration * 60

        return full_session_data