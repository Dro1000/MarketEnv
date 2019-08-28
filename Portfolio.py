
import gym
import numpy as np
import pandas as pd
import math 
from IPython.core.debugger import set_trace



class PortfolioEnv(gym.Env):
    def __init__(self, data, money = 1e+7, transaction_cost_pct=0.008, target="EURUSD"):
        """
        Here you can get data from financial market and then you can 
        create your market environment
        """
        self._kwargs = locals()
        #set_trace()
        self._data = pd.DataFrame.dropna(data)                        # data table
        self._target = target
        self._current_action = 0                 # the last action were done
        self._transaction_cost_pct = transaction_cost_pct      # transaction costs percentage 
        self._portfolio_value = money            # initial portfolio value
        self._money = money
        self._position_size = 1000000              # size of position; if positon is 0 then position_size = 0 
        
        data_pct_change = self._data.pct_change()
        state_space = pd.DataFrame.dropna(pd.concat([data_pct_change.shift(i) 
                            for i in range(9)], axis=1)).shift(0)
        roll = state_space.rolling(96)
        self._state_time_series = pd.DataFrame.dropna(
            (state_space - roll.mean()) / roll.std())
        
        self._dates = iter(self._state_time_series.index)      # dates 
        self._current_date = next(self._dates)    # current date
        self.state_shape = 6 + self._state_time_series.shape[1]
    
    def get_current_date(self):
        return self._current_date
    
    def get_current_portfolio_value(self):
        return self._portfolio_value
    
    def reset(self):
        self._dates = iter(self._state_time_series.index)      # dates 
        self._current_date = next(self._dates)    # current date
        self._portfolio_value = self._money
        self._current_action = 0 
        return self.render()
    
    def step(self, new_action=0, former_state=None):
        
        is_expl = former_state is None
        if is_expl:
            former_state = self.render()
            try:
                self._current_date = next(self._dates)
            except StopIteration:
                print("StopIteration occured!")
                self.__init__(**self._kwargs)
        
        d = self._position_size * abs(self._current_action - new_action) * self._transaction_cost_pct
        former_portfolio_value = self._portfolio_value
        
        new_portfolio_value = self._portfolio_value + new_action * self._position_size * \
                                 (self._data[[self._target]].diff().loc[self._current_date]) - d
        
        reward = np.log(new_portfolio_value / former_portfolio_value)
        
        if is_expl:
            self._portfolio_value = new_portfolio_value
            self._current_action = new_action

        return (self.render(), float(reward), False, former_state)
    
    def get_time_feature(self):
        date = self._current_date
        return np.sin(2 * math.pi * np.array([
            date.weekday() / 6, date.hour / 23, date.minute / 45]))
    
    def to_one_hot(self, action):
        one_hot = {-1: [1, 0, 0], 0: [0, 1, 0], 1:[0, 0, 1]}
        return(np.array(one_hot[action]))
    
        
    def render_(self):
        return(np.array(pd.concat([self._state_time_series.loc[[self._current_date]], 
                             pd.DataFrame({'action': self._current_action}, index=[self._current_date])], axis=1)))[0]
                            
    def render(self, action=None):
        
        if action == None:
            action = self._current_action
        
        return np.concatenate((self.get_time_feature(), 
                         np.array(self._state_time_series.loc[self._current_date]), 
                       self.to_one_hot(action)))
    
        
