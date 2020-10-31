import numpy as np

class SpatialDerivs:
    def __init__(self, direction: str, width: int, height: int):
        """Spacial Derivative object to deal with derivatives

        Args:
            direction: 'x' or 'y'
            width: the width of the space
            height: the height of the space

        """

        self.direction = direction
        self.width = width
        self.height = height
        self.derivs = np.zeros((height, width))
        self.last_exp_decay_t = 0

        self.decay_options = [None, 'exponential']
        self.edec = 0 # update exp decay every so iterations
        self.edecmax = 10

    def update(self, timestamp, x, y, polarity, norm=0.1, decay=None, exp_tau=0.1, exp_dt=0.001):
        """Update x derivatives using the kernel [-1, 1]

        Args:
            x: x value of the event
            y: y value of the event
            polarity: polarity of the event, 0 or 1
            timestamp: the timestamp of the event (in seconds)

        Keyword Args:
            norm: normalization constant multiplied by polarity
            decay: method of decaying derivatives over time (None, exponential)
            exp_tau: decay time constant, should be positive
            exp_dt: delay of when to calculate exponential decay

        Notes:
            - Exponential decay will run on the first event after exp_dt from the previous decay
        """

        if decay not in self.decay_options:
            raise ValueError("Invalid decay option")

        pol = (2 * polarity) - 1
        self.derivs[y, x] -= pol * 0.1
        if self.direction == 'x':
            self.derivs[y, max(x - 1, 0)] += pol * norm
        else:
            self.derivs[max(y - 1, 0), x] += pol * norm

        # Only update exponential decay every exp_dt to save computation
        if decay == 'exponential' and timestamp >= self.last_exp_decay_t + exp_dt:
            dt = timestamp - self.last_exp_decay_t
            self.derivs += (-1 / exp_tau) * self.derivs * dt
            """for i in range(self.derivs.shape[0]):
                for j in range(self.derivs.shape[1]):
                    #print('{}, {}, {}'.format(-1/exp_tau, self.derivs[i, j], dt))
                    #print((-1 / exp_tau) * self.derivs[i, j] * dt)
                    self.derivs[i, j] += (-1 / exp_tau) * self.derivs[i, j] * dt"""
            self.last_exp_decay_t = timestamp
            self.edec = 0
        self.edec += 1
        

