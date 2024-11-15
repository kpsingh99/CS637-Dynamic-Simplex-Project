import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import pickle
from pathlib import Path

class WeatherGenerator(nn.Module):
    """
    Generative model for sampling future weather parameters conditioned on current weather
    Uses a conditional VAE architecture for weather parameter generation
    """
    def __init__(self, input_dim: int = 5, latent_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 for current + previous weather
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),  # +input_dim for conditioning
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_var(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z, condition], dim=-1)
        return self.decoder(z)

    def sample(self, current_weather: torch.Tensor) -> torch.Tensor:
        """Sample next weather state given current weather"""
        with torch.no_grad():
            z = torch.randn(current_weather.size(0), self.latent_dim)
            return self.decode(z, current_weather)

class TrafficGenerator(nn.Module):
    """
    Generative model for sampling future traffic density conditioned on current density
    Uses a Gaussian mixture model for multi-modal predictions
    """
    def __init__(self, n_components: int = 3):
        super().__init__()
        self.n_components = n_components
        
        # Network to predict mixture parameters
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_components * 3)  # (weight, mean, std) for each component
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(x.unsqueeze(-1))
        weights, means, stds = torch.split(output, self.n_components, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        stds = torch.exp(stds)  # Ensure positive standard deviations
        return weights, means, stds

    def sample(self, current_density: torch.Tensor) -> torch.Tensor:
        """Sample next traffic density given current density"""
        with torch.no_grad():
            weights, means, stds = self(current_density)
            component = Categorical(weights).sample()
            return Normal(means[component], stds[component]).sample()

class SensorFailureGenerator:
    """
    Model for generating sensor failures based on weather conditions and historical data
    Uses empirical distributions and Markov chain for failure states
    """
    def __init__(self, transition_matrix: np.ndarray, failure_probs: Dict[str, float]):
        self.transition_matrix = transition_matrix  # State transition probabilities
        self.failure_probs = failure_probs  # Weather-conditioned failure probabilities
        self.current_state = None  # Current failure state
    
    def update_failure_probs(self, weather_params: Dict[str, float]):
        """Update failure probabilities based on current weather"""
        # Adjust probabilities based on weather conditions
        for condition, base_prob in self.failure_probs.items():
            if condition == "bright_image":
                self.failure_probs[condition] = base_prob * (1 + weather_params["sun_intensity"])
            elif condition == "blur":
                self.failure_probs[condition] = base_prob * (1 + weather_params["rain_intensity"])
    
    def sample_failure(self, weather_params: Dict[str, float]) -> Dict[str, bool]:
        """Sample sensor failures given weather conditions"""
        self.update_failure_probs(weather_params)
        failures = {}
        
        for sensor_type, prob in self.failure_probs.items():
            if self.current_state is None:
                # Initial failure sampling
                failures[sensor_type] = np.random.random() < prob
            else:
                # Use transition matrix for temporal consistency
                transition_prob = self.transition_matrix[int(self.current_state[sensor_type])]
                failures[sensor_type] = np.random.random() < transition_prob * prob
        
        self.current_state = failures
        return failures

class MonitorAlarmGenerator:
    """
    Generator for runtime monitor alarms using learned duration and arrival distributions
    """
    def __init__(self, historical_data_path: str):
        self.historical_data = pd.read_csv(historical_data_path)
        self.alarm_types = self.historical_data['alarm_type'].unique()
        
        # Fit duration and arrival time distributions
        self.duration_params = {}
        self.arrival_params = {}
        for alarm in self.alarm_types:
            alarm_data = self.historical_data[self.historical_data['alarm_type'] == alarm]
            self.duration_params[alarm] = {
                'mean': alarm_data['duration'].mean(),
                'std': alarm_data['duration'].std()
            }
            self.arrival_params[alarm] = {
                'rate': len(alarm_data) / len(self.historical_data)
            }
    
    def sample_alarms(self, time_window: float) -> List[Dict]:
        """Sample monitor alarms for given time window"""
        alarms = []
        
        for alarm_type in self.alarm_types:
            # Sample number of alarms using Poisson distribution
            n_alarms = np.random.poisson(self.arrival_params[alarm_type]['rate'] * time_window)
            
            for _ in range(n_alarms):
                # Sample arrival time uniformly in window
                arrival_time = np.random.uniform(0, time_window)
                
                # Sample duration from learned distribution
                duration = np.random.normal(
                    self.duration_params[alarm_type]['mean'],
                    self.duration_params[alarm_type]['std']
                )
                
                alarms.append({
                    'type': alarm_type,
                    'arrival_time': arrival_time,
                    'duration': max(0, duration)  # Ensure non-negative duration
                })
        
        return sorted(alarms, key=lambda x: x['arrival_time'])

class TrajectoryGenerator:
    """
    Main class that combines all generative models for trajectory sampling
    """
    def __init__(self, 
                 weather_model: WeatherGenerator,
                 traffic_model: TrafficGenerator,
                 sensor_model: SensorFailureGenerator,
                 alarm_model: MonitorAlarmGenerator):
        self.weather_model = weather_model
        self.traffic_model = traffic_model
        self.sensor_model = sensor_model
        self.alarm_model = alarm_model
    
    def sample_trajectory(self, 
                        current_state: Dict,
                        horizon: int = 10,
                        dt: float = 0.1) -> Dict:
        """
        Sample a complete trajectory including weather, traffic, sensor failures and alarms
        
        Args:
            current_state: Current system state including weather and traffic
            horizon: Number of time steps to predict
            dt: Time step duration in seconds
        """
        trajectory = {
            'weather': [],
            'traffic': [],
            'sensor_failures': [],
            'alarms': [],
            'timestamps': np.arange(horizon) * dt
        }
        
        # Convert current state to tensor
        current_weather = torch.tensor(current_state['weather']).float()
        current_traffic = torch.tensor(current_state['traffic']).float()
        
        # Generate trajectory
        for t in range(horizon):
            # Sample weather
            next_weather = self.weather_model.sample(current_weather)
            trajectory['weather'].append(next_weather.numpy())
            current_weather = next_weather
            
            # Sample traffic
            next_traffic = self.traffic_model.sample(current_traffic)
            trajectory['traffic'].append(next_traffic.numpy())
            current_traffic = next_traffic
            
            # Sample sensor failures
            weather_params = {
                'sun_intensity': next_weather[0].item(),
                'rain_intensity': next_weather[1].item()
            }
            failures = self.sensor_model.sample_failure(weather_params)
            trajectory['sensor_failures'].append(failures)
        
        # Sample alarms for entire trajectory
        trajectory['alarms'] = self.alarm_model.sample_alarms(horizon * dt)
        
        return trajectory

    @classmethod
    def load_models(cls, model_dir: str) -> 'TrajectoryGenerator':
        """Load all pretrained models from directory"""
        weather_model = WeatherGenerator()
        weather_model.load_state_dict(torch.load(f"{model_dir}/weather_model.pth"))
        
        traffic_model = TrafficGenerator()
        traffic_model.load_state_dict(torch.load(f"{model_dir}/traffic_model.pth"))
        
        with open(f"{model_dir}/sensor_model.pkl", 'rb') as f:
            sensor_model = pickle.load(f)
        
        alarm_model = MonitorAlarmGenerator(f"{model_dir}/historical_alarms.csv")
        
        return cls(weather_model, traffic_model, sensor_model, alarm_model)

def main():
    """Example usage of the trajectory generator"""
    # Load models
    generator = TrajectoryGenerator.load_models("path/to/models")
    
    # Define current state
    current_state = {
        'weather': [0.5, 0.2, 0.3, 0.1, 0.4],  # Example weather parameters
        'traffic': 0.3  # Current traffic density
    }
    
    # Sample trajectory
    trajectory = generator.sample_trajectory(current_state, horizon=20)
    
    # Print trajectory summary
    print("Sampled trajectory:")
    print(f"Weather transitions: {len(trajectory['weather'])}")
    print(f"Traffic changes: {len(trajectory['traffic'])}")
    print(f"Sensor failures: {sum(len(f) for f in trajectory['sensor_failures'])}")
    print(f"Monitor alarms: {len(trajectory['alarms'])}")

if __name__ == "__main__":
    main()