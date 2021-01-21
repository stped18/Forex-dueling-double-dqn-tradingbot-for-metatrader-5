class Agent(object):

    def __init__(self, epsilon=None):

        self.epsilon = epsilon

    def act(self, state):

        raise NotImplementedError()

    def observe(self, state, action, reward, next_state, terminal, *args):

        raise NotImplementedError()

    def end(self):

        pass