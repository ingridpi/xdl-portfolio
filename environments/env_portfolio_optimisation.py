import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class PortfolioOptimisationEnv(gym.Env):
    def __init__(
        self,
        df,
        stock_dimension,
        initial_amount,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        day=0,
    ):
        self.df = df
        self.day = day
        self.data = self.df.loc[self.day, :]
        self.stock_dim = stock_dimension
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling

        self.state_space = state_space
        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.state_space,
                self.stock_dim,
            ),
        )
        self.tech_indicator_list = tech_indicator_list

        # load data from a pandas dataframe
        self.state = [
            self.data[tech].values.tolist() for tech in self.tech_indicator_list
        ]
        self.terminal = False
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.episode = 0

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) /
            #                   (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data.close.values

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            current_day_memory = self.data.close.values
            self.state = [
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ]
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight

            portfolio_return = np.dot(
                ((current_day_memory / last_day_memory) - 1), weights
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = [
            self.data[tech].values.tolist() for tech in self.tech_indicator_list
        ]
        self.portfolio_value = self.initial_amount

        self.terminal = False
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.episode += 1
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date  # type: ignore
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
