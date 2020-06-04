class DummyAgent:
    def __init__(self, env):
        self.env = env
        self.env_action = self.env.action_space.sample()

    def action(self):
        self.env_action = self.env.action_space.sample()
        return self.env_action

    def reset(self):
        pass
