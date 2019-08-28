
import gym
import numpy as np
import pandas as pd

class marketEnv(gym.Env):
    def __init__(self, data):
        self._data = pd.DataFrame.dropna(data)
        self._dates = iter(self._data.index)
        self._current_date = next(self._dates)
    
    def get_current_date(self):
        return self._current_date
    
    def reset(self):
        self.__init__(self._data)
        return self.render()
    
    def render(self):
        return(self._data.loc[[self._current_date]])
