import torch
import numpy as np
from surrogate_model import SurrogateModel
from generative_model import TrajectoryGenerator
from typing import Dict

class MCTSNode:
    """
    A node in the MCTS tree representing a state and associated MCTS parameters.
    """
    def __init__(self, state, parent=None):
        self.state = state  # State representation at this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visit_count = 0  # Visit count for UCT
        self.value_sum = 0.0  # Cumulative value for this node
        self.is_fully_expanded = False  # Indicator if all actions have been tried
        self.untried_actions = self.get_possible_actions()  # Actions that can be tried from this state

    def get_possible_actions(self):
        # Define possible actions based on the environment or state attributes
        return ['action_1', 'action_2', 'action_3']  # Example actions
    
    def expand(self, action):
        # Expand this node with a new child based on the action taken
        new_state = self.simulate_action(action)
        child_node = MCTSNode(new_state, parent=self)
        self.children.append(child_node)
        self.untried_actions.remove(action)
        return child_node

    def simulate_action(self, action):
        # Placeholder: Define how an action affects the state
        # This should be replaced with actual action logic based on your environment
        return self.state  # Returning state directly here; modify as needed

    def is_leaf(self):
        # Check if node is a leaf node
        return len(self.children) == 0

    def update(self, value):
        # Update node statistics with the given value
        self.visit_count += 1
        self.value_sum += value

    def get_value(self):
        # Average value of this node
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0


class MCTS:
    def __init__(self, surrogate_model: SurrogateModel, trajectory_generator: TrajectoryGenerator, exploration_constant: float = 1.0):
        self.surrogate_model = surrogate_model
        self.trajectory_generator = trajectory_generator
        self.exploration_constant = exploration_constant

    def uct_value(self, node, parent_visit_count):
        """
        Calculate the UCT value for a node.
        """
        if node.visit_count == 0:
            return float('inf')  # Encourage exploration of new nodes
        exploitation_value = node.get_value()
        exploration_value = self.exploration_constant * np.sqrt(np.log(parent_visit_count) / node.visit_count)
        return exploitation_value + exploration_value

    def select_best_child(self, node):
        """
        Select the child with the highest UCT value.
        """
        parent_visit_count = node.visit_count
        return max(node.children, key=lambda child: self.uct_value(child, parent_visit_count))

    def rollout(self, current_state: Dict, horizon: int = 10) -> float:
        """
        Perform a rollout (simulation) starting from the current state.
        """
        trajectory = self.trajectory_generator.sample_trajectory(current_state, horizon=horizon)
        
        cumulative_reward = 0.0
        for state in trajectory['weather']:
            # Convert state to tensor and evaluate reward
            state_tensor = torch.tensor(state).float().unsqueeze(0)  # Convert to batch of 1 for model
            reward_info = self.surrogate_model(state_tensor)
            reward = reward_info['combined_reward'].item()
            cumulative_reward += reward
        
        return cumulative_reward

    def expand_and_simulate(self, node):
        """
        Expand the given node and perform a rollout simulation from the newly expanded node.
        """
        if not node.untried_actions:
            node.is_fully_expanded = True
            return 0.0  # No reward if there are no actions left to try
        
        # Select an action to expand
        action = node.untried_actions[0]  # Take the first untried action
        child_node = node.expand(action)
        
        # Perform a rollout simulation from this child node's state
        cumulative_reward = self.rollout(child_node.state)
        child_node.update(cumulative_reward)
        
        return cumulative_reward

    def backpropagate(self, node, reward):
        """
        Propagate the reward back up the tree.
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self, root_state: Dict, num_simulations: int = 1000) -> str:
        """
        Perform MCTS starting from the given root state.
        """
        root_node = MCTSNode(root_state)
        
        for _ in range(num_simulations):
            node = root_node
            
            # Selection: Traverse the tree to a leaf node
            while not node.is_leaf() and node.is_fully_expanded:
                node = self.select_best_child(node)
            
            # Expansion and Simulation
            reward = self.expand_and_simulate(node)
            
            # Backpropagation
            self.backpropagate(node, reward)
        
        # Return the best action from the root node
        best_child = max(root_node.children, key=lambda child: child.get_value())
        best_action = best_child  # Define based on your actions setup if needed
        return best_action