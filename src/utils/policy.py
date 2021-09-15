import numpy as np


class AdaptiveThresholdPolicy(object):

    def __init__(
            self,
            total_steps,
            lower_is_better=True,
            init_thres=1.0, 
            min_thres=0.1, 
            max_thres=1.0, 
            delta=0.01,
            window=10,
        ):
        super().__init__()
        self.min_thres = min_thres
        self.max_thres = max_thres
        self.delta = delta
        self.window = window
        self.thres = init_thres
        self.metrics = []
        self.total_steps = total_steps
        self.lower_is_better = lower_is_better

    def get_threshold(self, _):
        return self.thres  # independent of steps

    def get_window(self):
        return np.mean(self.metrics[-self.window:])

    def record(self, metric):
        record = self.get_window()
        if metric > record:
            if self.lower_is_better:
                # make the problem easier
                self.thres += self.delta
            else:
                # make problem harder
                self.thres -= self.delta
        elif metric < record:
            if self.lower_is_better:
                # make problem harder
                self.thres -= self.delta
            else:
                # make the problem easier
                self.thres += self.delta
        self.thres = max(self.min_thres, self.thres)
        self.thres = min(self.max_thres, self.thres)
        # record metric for next time
        self.metrics.append(metric)


class ConstantThresholdPolicy(object):

    def __init__(
            self,
            init_thres=1.0, 
        ):
        super().__init__()
        self.init_thres = init_thres

    def get_threshold(self, _):
        return self.init_thres


class LinearThresholdPolicy(object):

    def __init__(
            self,
            total_steps,
            max_anneal_step,
            init_thres=1.0, 
            min_thres=0.1, 
        ):
        super().__init__()

        self.schedule = np.concatenate([
            np.linspace(min_thres, init_thres, max_anneal_step)[::-1],  # start at 1.0
            np.ones(total_steps - max_anneal_step + 1) * min_thres,
        ])
        self.min_thres = min_thres
        self.thres = init_thres
        self.total_steps = total_steps

    def get_threshold(self, step):
        return self.schedule[step]


class StepThresholdPolicy(object):

    def __init__(
            self,
            total_steps,
            max_anneal_step,
            step_size=0.1,
            init_thres=1.0, 
            min_thres=0.1,
        ):
        super().__init__()
        schedule = np.arange(min_thres, init_thres+step_size, step_size)[::-1]
        schedule = np.repeat(schedule, int(max_anneal_step / len(schedule)))
        schedule = np.concatenate([
            schedule,
            np.ones(total_steps - max_anneal_step + 1) * min_thres
        ])
        self.schedule = schedule
        self.min_thres = min_thres
        self.thres = init_thres
        self.total_steps = total_steps

    def get_threshold(self, step):
        return self.schedule[step]


class ExponentialDecayThresholdPolicy(object):

    def __init__(
            self,
            total_steps,
            max_anneal_step,
            init_thres=1.0, 
            min_thres=0.1,
            decay=0.1
        ):
        super().__init__()
        schedule = (init_thres - min_thres) * np.exp(-decay*np.arange(max_anneal_step)) + min_thres
        schedule = np.concatenate([
            schedule,
            np.ones(total_steps - max_anneal_step + 1) * min_thres
        ])
        self.schedule = schedule
        self.min_thres = min_thres
        self.thres = init_thres
        self.total_steps = total_steps

    def get_threshold(self, step):
        return self.schedule[step]
