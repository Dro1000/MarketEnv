
import numpy as np

class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize. 
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)

        # add data to storage
        self._storage.append(data)
        if (self.__len__() > self._maxsize):
            del self._storage[0]
            
    def sample_sequence(self, sequence_size=96, batch_size=96):
        idxes = np.random.choice(range(self.__len__()), size=batch_size, replace=True)
        idx = np.random.choice(range(self.__len__() - batch_size + 1))
        idxes = np.arange(idx, idx + batch_size)
        
        # collect <s,a,r,s',done> for each index
        obs_batch = [self._storage[i][0] for i in idxes]
        act_batch = [self._storage[i][1] for i in idxes]
        rew_batch = [self._storage[i][2] for i in idxes]
        next_obs  = [self._storage[i][3] for i in idxes]
        done_mask = [self._storage[i][4] for i in idxes]
        
        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch), \
               np.array(next_obs), np.array(done_mask)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = np.random.choice(range(self.__len__()), size=batch_size, replace=True)
        
        # collect <s,a,r,s',done> for each index
        obs_batch = [self._storage[i][0] for i in idxes]
        act_batch = [self._storage[i][1] for i in idxes]
        rew_batch = [self._storage[i][2] for i in idxes]
        next_obs  = [self._storage[i][3] for i in idxes]
        done_mask = [self._storage[i][4] for i in idxes]
        
        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch), \
               np.array(next_obs), np.array(done_mask)
