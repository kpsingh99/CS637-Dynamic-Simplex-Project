import torch
import torch.nn as nn

class SurrogateModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        
        # Shared encoder for state information
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Performance score predictor (λp) - predicts average speed
        self.performance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()  # Normalize speed score
        )
        
        # Safety score predictor (λf) - predicts collision likelihood
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()  # Probability of collision
        )
        
        # Additional cost predictor (λc) if needed
        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        # Hyperparameters for reward combination
        self.alpha1 = nn.Parameter(torch.tensor(1.0))  # Performance weight
        self.alpha2 = nn.Parameter(torch.tensor(1.0))  # Safety weight
        self.alpha3 = nn.Parameter(torch.tensor(1.0))  # Cost weight

    def forward(self, state):
        """
        Args:
            state: Tensor containing state information including:
                  - Current weather parameters
                  - Traffic density
                  - Sensor status
                  - Monitor states
                  - Vehicle state (speed, position, etc.)
        Returns:
            Dictionary containing:
                - performance_score (λp)
                - safety_score (λf)
                - cost_score (λc)
                - combined_reward (R)
        """
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Predict individual scores
        performance_score = self.performance_head(encoded_state)
        safety_score = self.safety_head(encoded_state)
        cost_score = self.cost_head(encoded_state)
        
        # Calculate combined reward
        # R(st,a) = α1·λp(st,a) - α2·λf(st,a) - α3·λc(st,a)
        reward = (self.alpha1 * performance_score - 
                 self.alpha2 * safety_score - 
                 self.alpha3 * cost_score)
        
        return {
            'performance_score': performance_score,
            'safety_score': safety_score,
            'cost_score': cost_score,
            'combined_reward': reward
        }

class GenerativeWeatherModel(nn.Module):
    """Model for predicting weather parameter transitions"""
    def __init__(self, weather_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, weather_dim)
        )
    
    def forward(self, current_weather):
        return self.net(current_weather)

class TrafficDensityModel(nn.Module):
    """Model for predicting traffic density changes"""
    def __init__(self, density_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(density_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, density_dim)
        )
    
    def forward(self, current_density):
        return self.net(current_density)

class SensorFailureModel(nn.Module):
    """Model for predicting sensor failures based on conditions"""
    def __init__(self, condition_dim, n_sensors, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_sensors),
            nn.Sigmoid()  # Probability of failure for each sensor
        )
    
    def forward(self, conditions):
        return self.net(conditions)