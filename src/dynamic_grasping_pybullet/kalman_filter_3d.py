import numpy as np


# adapted from https://github.com/zziz/kalman-filter and https://gist.github.com/manicai/922976

class KalmanFilter(object):
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):

        if (F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F(0).shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, dt, u=0, predict_only=False):
        x = np.dot(self.F(dt), self.x) + np.dot(self.B, u)
        P = np.dot(np.dot(self.F(dt), self.P), self.F(dt).T) + self.Q
        if not predict_only:
            self.x = x
            self.P = P
        return x

    def update(self, z):
        """
        :param z: (ndim, 1) np array
        """
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


class RealWorld:
    def __init__(self, ndim=3):
        self.position = np.array([0.0] * ndim)
        self.velocity = np.array([0.5] * ndim)
        self.time_step = 0.1
        self.time = 0.0
        self.measure = None

        # Noise on the measurements
        self.measurement_variance = 0.002

        # If we want to kink the profile.
        self.change_after = 50
        self.changed_velocity = -0.5

    def measurement(self):
        if self.measure == None:
            self.measure = (self.position
                            + np.random.normal(0, self.measurement_variance, size=len(self.position)))
        return self.measure

    def step(self):
        self.time += self.time_step
        self.position += self.velocity * self.time_step

        if self.time >= self.change_after:
            self.velocity = self.changed_velocity
        self.measure = None


def example_ND(ndim=3):
    world = RealWorld()
    dt = world.time_step

    # state transition model
    def F_func(dt):
        F = np.eye(3 * ndim)
        F[0:ndim, ndim:2 * ndim] = np.eye(ndim) * dt
        F[ndim:2 * ndim, 2 * ndim:3 * ndim] = np.eye(ndim) * dt
        F[0:ndim, 2 * ndim:3 * ndim] = np.eye(ndim) * dt * dt / 2
        return F

    # observation model
    H = np.zeros((ndim, 3 * ndim))
    H[:ndim, :ndim] = np.eye(ndim)

    # the covariance of the process noise
    Q = np.zeros_like(F_func(0))
    Q[:2 * ndim, :2 * ndim] = 0.05

    # the covariance of the observation noise
    R = np.eye(ndim) * 0.5

    kf = KalmanFilter(F=F_func, H=H, Q=Q, R=R)
    measurements = []
    predictions = []
    predictions_future = []
    future_horizon = 10

    for _ in range(1000):
        world.step()
        measurement = world.measurement()
        measurements.append(measurement)

        predictions.append(np.dot(H, kf.predict(dt=dt))[0])
        predictions_future.append(np.dot(H, kf.predict(dt=dt * future_horizon, predict_only=True))[0])
        kf.update(measurement[:, None])

    import matplotlib.pyplot as plt
    plt.plot(range(len(measurements)), np.array(measurements)[:, 0], label='Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label='Kalman Filter Prediction')
    plt.plot(range(future_horizon, future_horizon + len(predictions_future)), np.array(predictions_future),
             label='Kalman Filter Future Prediction')
    plt.legend()
    plt.show()


def create_kalman_filter(x0=None, ndim=3):

    # state transition model
    def F_func(dt):
        F = np.eye(3 * ndim)
        F[0:ndim, ndim:2 * ndim] = np.eye(ndim) * dt
        F[ndim:2 * ndim, 2 * ndim:3 * ndim] = np.eye(ndim) * dt
        F[0:ndim, 2 * ndim:3 * ndim] = np.eye(ndim) * dt * dt / 2
        return F

    # observation model
    H = np.zeros((ndim, 3 * ndim))
    H[:ndim, :ndim] = np.eye(ndim)

    # the covariance of the process noise
    Q = np.zeros_like(F_func(0))
    Q[:2 * ndim, :2 * ndim] = 0.05

    # the covariance of the observation noise
    R = np.eye(ndim) * 0.5

    kf = KalmanFilter(F=F_func, H=H, Q=Q, R=R, x0=x0)
    return kf


if __name__ == '__main__':
    example_ND()