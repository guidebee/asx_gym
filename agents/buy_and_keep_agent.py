from asx_gym.envs import BUY_STOCK, HOLD_STOCK
from asx_gym.envs import AsxAction, AsxTransaction


class BuyAndKeepAgent:
    def __init__(self, env, company_id, min_volume=10, max_volume=100):
        self.env = env
        self.env_action = self.env.action_space.sample()
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.company_id = company_id
        self.asx_hold_action = AsxAction(1)
        asx_transaction = AsxTransaction(company_id,
                                         HOLD_STOCK, 0, 0)
        self.asx_hold_action.add_transaction(asx_transaction)
        self.bought_stock = False

    def action(self):
        if not self.bought_stock:
            self.bought_stock = True
            asx_action = AsxAction(1)
            asx_transaction = AsxTransaction(self.company_id,
                                             BUY_STOCK, 0, 10000)
            asx_action.add_transaction(asx_transaction)
        else:
            asx_action = self.asx_hold_action

        asx_action.copy_to_env_action(self.env_action)
        return self.env_action

    def reset(self):
        self.bought_stock = False
