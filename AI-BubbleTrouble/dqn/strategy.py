import math


class EpsilonGreedyStrategy:
    """
    This class responsible for providing the exploration - exploitation strategy
    """

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, curr_step):
        """
        :param curr_step: the current step number
        :return: exploration rate
        """
        return self.end + (self.start - self.end) * math.exp(-curr_step / self.decay)
