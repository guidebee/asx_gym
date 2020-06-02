import random

from asx_gym.envs import BUY_STOCK, SELL_STOCK, HOLD_STOCK
from asx_gym.envs import AsxAction, AsxTransaction


class RandomAgent:
    def __init__(self, env, min_volume=10, max_volume=100):
        self.env = env
        self.env_action = self.env.action_space.sample()
        self.min_volume = min_volume
        self.max_volume = max_volume

    def action(self):
        simulate_company_list = self.env.simulate_company_list
        company_count = len(simulate_company_list)
        self.env_action['company_count'] = company_count
        end = random.randint(0, 10)
        if end > 7:
            end_batch = 1
        else:
            end_batch = 0

        asx_action = AsxAction(end_batch)
        for c in range(len(simulate_company_list)):
            bet = random.randint(0, 10)
            company_id_index = random.randint(0, company_count) % company_count
            if bet > 7:
                stock_operation = BUY_STOCK
                price = 1000
            elif bet > 2:
                stock_operation = SELL_STOCK
                price = 1.0
            else:
                stock_operation = HOLD_STOCK
                price = 0
            volume = random.randint(self.min_volume, self.max_volume)
            asx_transaction = AsxTransaction(simulate_company_list[company_id_index],
                                             stock_operation, volume, price)
            asx_action.add_transaction(asx_transaction)

        asx_action.copy_to_env_action(self.env_action)
        return self.env_action
