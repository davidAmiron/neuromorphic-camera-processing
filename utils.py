import numpy as np
import cv2
from sklearn import linear_model

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
        self.last_decay_t = 0
        self.to_remove = []

        self.optical_flow = OpticalFlow(self.width, self.height)

        self.decay_options = [None, 'exponential', 'constant', 'lifetime']

    def update(self, timestamp, x, y, polarity, norm=0.1, decay=None, exp_tau=0.1,
               delay_dt=0.001, const_event_len=3e-2):
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
            decay_dt: delay of when to calculate exponential decay, in seconds
            const_event_len: when constant decay is selected, the number of milliseconds an event is active

        Notes:
            - Exponential decay will run on the first event after delay_dt from the previous decay
        """

        if decay not in self.decay_options:
            raise ValueError("Invalid decay option")

        # Polarity is (0, 1), change it to (-1, 1)
        pol = (2 * polarity) - 1

        self.derivs[y, x] -= pol * norm
        if self.direction == 'x':
            self.derivs[y, max(x - 1, 0)] += pol * norm
        else:
            self.derivs[max(y - 1, 0), x] += pol * norm


        # Assign event removal times for contant and lifetime estimation
        if decay == 'constant':
            self.to_remove.append({'time': timestamp, 'y': y, 'x': x, 'delta': pol * norm})
            if self.direction == 'x':
                self.to_remove.append({'time': timestamp, 'y': y, 'x': max(x - 1, 0), 'delta': -pol * norm})
            else:
                self.to_remove.append({'time': timestamp, 'y': max(y - 1, 0), 'x': x, 'delta': -pol * norm})
        elif decay == 'lifetime':
            self.optical_flow.update(timestamp, x, y, polarity)
            lifetime = 1
            #self.to_remove.append({'

        # Only update exponential decay every delay_dt to save computation
        if decay == 'exponential' and timestamp >= self.last_decay_t + delay_dt:
            dt = timestamp - self.last_decay_t
            self.derivs += (-1 / exp_tau) * self.derivs * dt
            self.last_decay_t = timestamp

        elif decay == 'constant' and timestamp >= self.last_decay_t + const_event_len:
            self.last_decay_t = timestamp
            next_to_remove = []
            for event in self.to_remove:
                if event['time'] < timestamp:
                    self.derivs[event['y'], event['x']] += event['delta']
                else:
                    next_to_remove.append(event)
            self.to_remove = next_to_remove

            

def draw_gradients(image, x_derivs, y_derivs, x_scale=50, y_scale=50):
    assert(x_derivs.shape == y_derivs.shape)
    height = x_derivs.shape[0]
    width = x_derivs.shape[1]
    for h in range(0, height, 5):
        for w in range(0, width, 5):
            start_point = (h, w)
            end_point = (int(h + y_derivs[h, w] * y_scale), int(w + x_derivs[h, w] * x_scale))
            image = cv2.arrowedLine(image, start_point, end_point, 255, 1)
    return image

        
class OpticalFlow:
    def __init__(self, width: int, height: int, window_t: float = 5, window_l: int = 5):
        """OpticalFlow object to calculate optical flow of spiking image

        Args:
            width: the width of the space
            height: the height of the space
            window_t: time in milliseconds of window to use for plane calculation
            window_l: length in pixels of box for plane calculations
        """
        self.width = width
        self.height = height
        self.window_t = window_t
        self.window_l = window_l
        self.flows = np.zeros((height, width, 2))
        self.sae = np.zeros((height, width))

        self.calc_spacing = 10  # Calculate every 10 events, to save time
        self.curr_event = 0
        self.num_fail = 0
        self.total = 0

    def draw_flows(self, img, scale=1e-2, skip=5, normalize=True, norm_scale=5):
        for y in range(0, self.height, skip):
            for x in range(0, self.width, skip):
                fx = self.flows[y, x, 0]
                fy = self.flows[y, x, 1]
                if fy == 0.0 or fx == 0.0:
                    continue
                start_point = (y, x)
                delta = np.array([fy, fx])
                if normalize:
                    print(delta)
                    delta /= np.linalg.norm(delta)
                    delta *= norm_scale
                else:
                    delta *= scale
                print(delta)
                end_point = delta + np.array([y, x])
                end_point = tuple(end_point.astype(int))
                print(end_point)
                img = cv2.arrowedLine(img, start_point, end_point, 255, 1)
        return img

    def update(self, timestamp, x, y, polarity):
        """Update optical flow vector field

        Look at only positive events??
        """
        if polarity == 1:
            return
        
        self.total += 1

        self.sae[y, x] = timestamp
        if self.curr_event == self.calc_spacing:
            self.curr_event = 0
            
            # Get events in window
            r = self.window_l // 2
            ym = y - r
            yp = y + r
            xm = x - r
            xp = x + r
            ym = np.clip(ym, 0, self.height)
            yp = np.clip(yp, 0, self.height)
            xm = np.clip(xm, 0, self.width)
            xp = np.clip(xp, 0, self.width)
            xr = np.arange(xm, xp + 1)
            yr = np.arange(ym, yp + 1)
            xmesh, ymesh = np.meshgrid(xr, yr)
            points = np.vstack([xmesh.ravel(), ymesh.ravel()]).T
            times = self.sae[ym:yp+1,xm:xp+1]

            # Fit plane to window
            ransac = linear_model.RANSACRegressor()
            try:
                ransac.fit(points, times.ravel())

                # Get fit coefficients
                cx = ransac.estimator_.coef_[0]
                cy = ransac.estimator_.coef_[1]
                print(cx)
                print(cy)
                print()
                if cx != 0 and cy != 0:
                    dxdt = 1 / ransac.estimator_.coef_[0]
                    dydt = 1 / ransac.estimator_.coef_[1]
                    self.flows[y, x, 0] = dxdt
                    self.flows[y, x, 1] = dydt
            except ValueError as v:
                self.num_fail += 1
            
            
        else:
            self.curr_event += 1

