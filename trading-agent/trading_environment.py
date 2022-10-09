import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding



class DataSource:
    def __init__(self, features, data_path='data/data_train.csv', ticker='AAPL', trading_days=252):
        self.features = features
        self.data_path = data_path
        self.ticker = ticker
        self.trading_days = trading_days
        self.data = self.load_data()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.offset = None
        self.dates = self.data.index
        

    def load_data(self):
        data = pd.read_csv(self.data_path, usecols=["Date","ticker","reward"]+self.features)
        data = data[data['ticker']==self.ticker]
        data = data.drop(['ticker'], axis=1)
        data = data.set_index("Date")
        return data

    def reset(self):
        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days
        return obs[0], obs[1:], done


class TradingSimulator:
    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        end_position = action - 1  # short, neutral, long
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[self.step]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)




class TradingEnvironment(gym.Env):
    def __init__(self,
                 ticker='AAPL',
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 data_path='data/data_nosentiment_train.csv',
                 features=[
                    'returns_1', 'returns_2', 'returns_5', 'returns_10', 'returns_21',
                    'STOCH', 'ULTOSC', 'RSI', 'MACD', 'ATR',
                    'count_news','count_opinions',
                    'tweetCount_cashtags','retweetCount_cashtags','tweetCount_keywords','retweetCount_keywords',
                    'sent_content_news_1', 'sent_content_news_5','sent_content_news_21',
                    'sent_title_news_1', 'sent_title_news_5','sent_title_news_21',
                    'sent_content_opinions_1','sent_content_opinions_5', 'sent_content_opinions_21',
                    'sent_title_opinions_1', 'sent_title_opinions_5','sent_title_opinions_21',
                    'sent_tweeteval_keywords_1','sent_tweeteval_keywords_5', 'sent_tweeteval_keywords_21',
                    'sent_tweeteval_cashtags_1', 'sent_tweeteval_cashtags_5','sent_tweeteval_cashtags_21'
                 ]
                ):
        self.features = features
        self.ticker = ticker
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.data_path = data_path
        self.data_source = DataSource(features=self.features,
                                      data_path=self.data_path, 
                                      ticker=self.ticker, 
                                      trading_days=self.trading_days
                                     )


        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps)
        

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.float32(self.data_source.min_values),
                                            np.float32(self.data_source.max_values))
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        market_return, features, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                market_return=market_return)
        return features, reward, done, info #pridal jsem market_return

    def reset(self):
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[1]
