#!/usr/bin/env python

import unittest
from functools import partial

import gymnasium as gym

from spinup import ppo_tf1 as ppo


class TestPPO(unittest.TestCase):
    def test_cartpole(self):
        ''' Test training a small agent in a simple environment '''
        env_fn = partial(gym.make, 'CartPole-v1')
        ac_kwargs = dict(hidden_sizes=(32,))



if __name__ == '__main__':
    unittest.main()
