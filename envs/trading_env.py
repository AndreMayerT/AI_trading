import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, lookback_window_size=5, commission=10, initial_balance=1000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.open_prices = df["Abertura"].values
        self.close_prices = df["Fechamento"].values
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.comission = commission
        # Actions: 0 = hold, 1 = long, 2 = short, 3 = close.
        self.action_space = spaces.Discrete(4)
        # Observes the OHCLV values, net worth, and trade history
        self.observation_space = spaces.Box(
            low=self.df["Fechamento"].min(),
            high=self.df["Fechamento"].max(),
            shape=(5, lookback_window_size - 1),
            dtype=np.float16,
        )

    def reset(self):
        self.balance = self.initial_balance
        self.action_step = 0
        self.open_position = -1
        self.current_step = self.lookback_window_size - 1
        return self._next_observation(), []

    def _next_observation(self):
        start = self.current_step - self.lookback_window_size + 1
        obs = np.array(
            [
                self.df["Abertura"].values[start : self.current_step],
                self.df["Maxima"].values[start : self.current_step],
                self.df["Minima"].values[start : self.current_step],
                self.df["Fechamento"].values[start : self.current_step],
                self.df["VWAP D"].values[start : self.current_step],
            ]
        )
        return obs

    def step(self, action):
        print(action)
        self._take_action(action)
        self.current_step += 1
        obs = self._next_observation()
        reward = self._current_position()

        done = self.balance <= 0
        return obs, reward, done, False, {}

    def _current_position(self):
        if self.open_position == 1:
            current_position_result = int(
                (
                    self.close_prices[self.current_step]
                    - self.open_prices[self.action_step]
                )
                * 10
            )
        elif self.open_position == 2:
            current_position_result = int(
                (
                    -(
                        self.close_prices[self.current_step]
                        - self.open_prices[self.action_step]
                    )
                )
                * 10
            )
        else:
            current_position_result = 0

        if self.balance + current_position_result <= 0:
            self.balance = 0

        return current_position_result

    def _take_action(self, action):
        if action == 1:
            if self.open_position == -1:
                self.balance -= self.comission
                self.action_step = self.current_step
                self.open_position = 1
            elif self.open_position == 2:
                self.balance += self._current_position() - self.comission
                self.action_step = self.current_step
                self.open_position = 1

        elif action == 2:
            if self.open_position == -1:
                self.balance -= self.comission
                self.action_step = self.current_step
                self.open_position = 2
            elif self.open_position == 1:
                self.balance += self._current_position() - self.comission
                self.action_step = self.current_step
                self.open_position = 2

        elif action == 3:
            if self.open_position == 1 or self.open_position == 2:
                self.balance += self._current_position()
                self.open_position = -1

    def render(self, mode="human", close=False):
        open_position_to_string = {-1: "None", 1: "Long", 2: "Short"}
        print("Current position: " + open_position_to_string[self.open_position])
        print(f"Current operation: {self._current_position()}")
        print(f"Current balance: {self.balance}")
        print(f"Current step: {self.current_step}")
        print(
            "-------------------------------------------------------------------------"
        )
        start = 0
        if self.current_step > 50:
            start = self.current_step - 50
        if self.open_position == -1:
            plt.plot(
                [self.current_step - 1, self.current_step],
                self.open_prices[self.current_step - 1 : self.current_step + 1],
                color="black",
            )
        if self.open_position == 1:
            plt.plot(
                [self.current_step - 1, self.current_step],
                self.open_prices[self.current_step - 1 : self.current_step + 1],
                color="green",
            )
        if self.open_position == 2:
            plt.plot(
                [self.current_step - 1, self.current_step],
                self.open_prices[self.current_step - 1 : self.current_step + 1],
                color="red",
            )
        plt.pause(0.01)
