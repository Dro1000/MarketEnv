
from IPython.core.debugger import set_trace
from torch import nn
import torch
from torch.distributions.normal import Normal
import numpy as np
import re

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0, device=torch.device('cuda')):

        super().__init__()
        self.device=device
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        # Define your network body here. Please make sure agent is fully contained here
        assert len(state_shape) == 1
        state_dim = state_shape[0]
        
        self.nn = nn.Sequential()
        self.nn.add_module('layer1', nn.Linear(state_shape[0], 64))
        self.nn.add_module('relu1', nn.ELU())
        self.nn.add_module('dropout1', nn.Dropout(0.2))
        
        self.nn.add_module('layer2', nn.Linear(64, 64))
        self.nn.add_module('relu2', nn.ELU())
        self.nn.add_module('dropout2', nn.Dropout(0.4))
        
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.output = nn.Sequential(nn.Linear(64, 3), nn.ELU())
        
        self.init_weights()
        
        self.hidden = torch.zeros(64, device=device)
        self.cell_state = torch.zeros(64, device=device)
        
    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state
        input_to_lstm = self.nn(torch.tensor(state_t))
        
        hiddens, state_states = self.lstm(input_to_lstm[None, :, :])
        
        qvalues = self.output(hiddens[0])

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
    
    def update_weights(self, agent_weights, lr=0.0001):
        """
        This method update weights of target_network with another agent
        via formula: Q'(s, t) = Q'(s, t) + lr * Q(s, t), 
        where Q' - target network, 
        Q - agent
        
        Params:
        agent: agent network
        lr: learning rate
        """
        
        target_weights = self.state_dict()
        new_weights = {}
        
        for key in target_weights:
            new_weights[key] = (1 - lr) * target_weights[key] + lr * agent_weights[key]
            
        self.load_state_dict(new_weights)
        
    def init_weights(self):
        """
        This method initialize weights
        """
        weights = self.state_dict()
        for key, value in weights.items():
            split_key = re.split("_|\.", key)
            if 'bias' in split_key:
                if ('hh' in split_key):
                    weights[key] = torch.ones(value.size())
                else:
                    weights[key] = torch.zeros(value.size())
            elif "output" in split_key:
                weights[key] = Normal(loc=0, scale=0.001).sample(weights[key].size())
            else:
                weights[key] = torch.eye(weights[key].size()[0], weights[key].size()[1])
                
        self.load_state_dict(weights)
            
    
        
