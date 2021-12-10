import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import argparse
from collections import deque
import datetime
import os


def get_object_trajectory(length, speed, distance, theta, sampling_frequency=240., is_sinusoid=True):
    num_steps = int(length / speed * sampling_frequency)
    start_position = np.array([0, -length / 2.0, 1])
    target_position = np.array([0, length / 2.0, 1])
    position_trajectory = np.linspace(start_position, target_position, num_steps)
    if is_sinusoid:
        amplitude = distance / 2.
        period = (length / 3)
        # period = np.random.uniform(low=(length / 3), high=(length / 2))
        position_trajectory[:, 0] = amplitude * np.sin(2 * np.pi * position_trajectory[:, 1] / period)
    T_1 = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    T_2 = np.array([[1, 0, distance * np.cos(theta)], [0, 1, distance * np.sin(theta)], [0, 0, 1]])
    position_trajectory = np.dot(T_2, np.dot(T_1, position_trajectory.T)).T
    return position_trajectory


def create_lstm_model(input_shape, output_shape, sequence_len=None, stateful=False, batch_sz=None):
    simple_lstm_model = tfk.models.Sequential([
        tfkl.Input(batch_shape=(batch_sz, sequence_len,) + input_shape),
        tfkl.Reshape((-1, np.prod(input_shape))),
        tfkl.LSTM(100, return_sequences=True, stateful=stateful),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dropout(rate=0.1),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(np.prod(output_shape)),
        tfkl.Reshape((-1,) + output_shape),
    ])
    return simple_lstm_model


def create_mlp_model(input_shape, output_shape):
    simple_mlp_model = tfk.models.Sequential([
        tfkl.Input(shape=(input_shape)),
        tfkl.Flatten(),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dropout(rate=0.1),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(np.prod(output_shape)),
        tfkl.Reshape(output_shape),
    ])
    return simple_mlp_model


def create_dataset(num_data_points=5, sampling_frequency=240., is_sinusoid=True, distance_low=0.15, distance_high=0.4,
                   length=1.0, speeds=(0.01, 0.03, 0.05)):

    distance_to_robot_list = np.random.uniform(low=distance_low, high=distance_high, size=num_data_points)
    theta_list = np.random.uniform(low=0, high=360, size=num_data_points) * np.pi / 180
    speed_list = np.random.choice(speeds, size=num_data_points)

    trajectories = []

    for speed, distance, theta in zip(speed_list, distance_to_robot_list, theta_list):
        position_trajectory = get_object_trajectory(length, speed, distance, theta, sampling_frequency,
                                                    is_sinusoid=is_sinusoid)
        trajectories.append(position_trajectory)

    return trajectories


def create_dataset_from_trajectories(trajectories, data_gen_sampling_frequency=240.,
                                     measurement_sampling_frequency=240., future_horizons=(1.,), history=3):
    # future horizon in seconds, how far into the future to predict
    # history: how many past measurements to use for prediction

    subsample_ratio = int(data_gen_sampling_frequency / measurement_sampling_frequency)
    future_index_skip = (np.array(future_horizons) * data_gen_sampling_frequency).astype(int)

    X, Y = [], []
    for traj in trajectories:
        for idx in range(history*subsample_ratio, len(traj) - max(future_index_skip), subsample_ratio):
            x, y = traj[idx-history*subsample_ratio: idx: subsample_ratio], traj[idx+future_index_skip]
            X.append(x)
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


class DataGenerator(object):

    def __init__(self, trajectories, data_gen_sampling_frequency=240., measurement_sampling_frequency=240.,
                 future_horizons=(1.,), history=3):

        self.trajectories = trajectories
        self.min_traj_len = min([len(traj ) for traj in self.trajectories])
        self.data_gen_sampling_frequency = data_gen_sampling_frequency
        self.measurement_sampling_frequency = measurement_sampling_frequency
        self.future_horizons = future_horizons
        self.history = history
        self.input_shape = (history, trajectories[0].shape[-1])
        self.output_shape = (len(future_horizons), trajectories[0].shape[-1])

        self.subsample_ratio = int(data_gen_sampling_frequency / measurement_sampling_frequency)
        self.future_index = (np.array(future_horizons) * data_gen_sampling_frequency).astype(int)

    def generate(self, batch_sz):
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            # indexes = np.arange(len(self.trajectories))
            indexes = np.random.permutation(len(self.trajectories))

            # Generate batches
            for start_idx in range(0, len(indexes) - batch_sz, batch_sz):
                X, Y = np.zeros((batch_sz,) + self.input_shape), np.zeros((batch_sz,) + self.output_shape)
                for b_id, traj_id in enumerate(range(start_idx, start_idx + batch_sz)):
                    traj = self.trajectories[traj_id]
                    # randomly pick region within trajectory
                    idx = np.random.randint(self.history * self.subsample_ratio, len(traj) - max(self.future_index))
                    X[b_id] = traj[idx - self.history * self.subsample_ratio: idx: self.subsample_ratio]
                    Y[b_id] = traj[idx + self.future_index]

                # preprocess data e.g. make first point the reference
                Y -= X[:, 0:1, :]
                X -= X[:, 0:1, :]
                # X[:, 1:, :] = np.diff(X[0], axis=-2)
                yield X, Y

    def generate_sequence(self, batch_sz):
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            # indexes = np.arange(len(self.trajectories))
            indexes = np.random.permutation(len(self.trajectories))

            # Generate batches
            num_divs = (self.min_traj_len - max(self.future_index) - self.history * self.subsample_ratio) // self.subsample_ratio
            for start_idx in range(0, len(indexes) - batch_sz, batch_sz):
                X = np.zeros((batch_sz, num_divs) + self.input_shape)
                Y = np.zeros((batch_sz, num_divs) + self.output_shape)
                for b_id, traj_id in enumerate(range(start_idx, start_idx + batch_sz)):
                    traj = self.trajectories[traj_id]
                    start_point = np.random.randint(max(len(traj) - self.min_traj_len, 1))
                    # randomly pick region within trajectory
                    for seq_id, idx in enumerate(range(start_point+ self.history * self.subsample_ratio, start_point + self.min_traj_len - max(self.future_index), self.subsample_ratio)):
                        # idx = np.random.randint(self.history * self.subsample_ratio, len(traj) - max(self.future_index))
                        X[b_id, seq_id] = traj[idx - self.history * self.subsample_ratio: idx: self.subsample_ratio]
                        Y[b_id, seq_id] = traj[idx + self.future_index]

                # preprocess data e.g. make first point the reference
                Y -= X[:, :, 0:1, :]
                X -= X[:, :, 0:1, :]
                # X[:, 1:, :] = np.diff(X[0], axis=-2)  # not quite
                yield X, Y


def test_lstm_model(prediction_model, test_traj=None, args=None):

    # test_traj = trajectories[:1]
    X, Y = create_dataset_from_trajectories(test_traj, data_gen_sampling_frequency=args.data_gen_sampling_frequency,
                                            measurement_sampling_frequency=args.measurement_sampling_frequency,
                                            future_horizons=args.future_horizons, history=args.history)
    print('X.shape: {}, \t Y.shape: {}'.format(X.shape, Y.shape))
    # import ipdb; ipdb.set_trace()
    # # plt.plot(*trajectories[0][:, :2].T)
    # for i in range(0, len(X), 200):
    #     plt.plot(*test_traj[0][:, :2].T)
    #     plt.scatter(*X[i][:, :2].T, color='r')
    #     plt.scatter(*Y[i][:, :2].T, color='b')
    #     plt.show()

    predictions = []
    prediction_model.reset_states()
    for i in range(len(X)):
        # keep state info until reset
        Y_pred = prediction_model.predict((X[i] - X[i][:1, :])[None, None, ...]).squeeze() + X[i][:1, :]
        predictions.append(Y_pred)

        # plt.clf()
        # plt.plot(*test_traj[0][:, :2].T)
        # plt.scatter(*X[i][:, :2].T, color='r')
        # plt.scatter(*Y[i][:, :2].T, color='b')
        # plt.scatter(*Y_pred.squeeze()[:, :2].T, color='g')
        # # plt.show()
        # plt.draw()
        # plt.pause(0.001)

    Y_pred = np.array(predictions)
    for j in range(Y.shape[1]):
        for i in range(0, Y.shape[0], 20):
            plt.plot([Y[i, j, 0], Y_pred[i, j, 0]], [Y[i, j, 1], Y_pred[i, j, 1]])
        plt.title('Error: {}'.format(np.mean(np.linalg.norm(Y[:, j, :] - Y_pred[:, j, :], axis=1))))
        plt.show()
    plt.plot(*trajectories[0][:, :2].T)
    import ipdb; ipdb.set_trace()


def train_lstm_model(data_gen, trajectories, args):
    data_gen_val = DataGenerator(trajectories, data_gen_sampling_frequency=args.data_gen_sampling_frequency,
                                 measurement_sampling_frequency=args.measurement_sampling_frequency,
                                 future_horizons=args.future_horizons, history=args.history)
    # sample = data_gen_val.generate_sequence(2).__next__()
    simple_lstm_model = create_lstm_model(input_shape=data_gen.input_shape, output_shape=data_gen.output_shape)
    simple_lstm_model.compile('adam', 'mean_squared_error')

    log_dir = "logs/" + datetime.datetime.now().strftime("lstm_model_%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    training_history = simple_lstm_model.fit(data_gen.generate_sequence(args.batch_sz), epochs=args.epochs,
                                             steps_per_epoch=5 * len(trajectories) / args.batch_sz,
                                             validation_data=data_gen_val.generate_sequence(args.batch_sz), validation_steps=10,
                                             callbacks=[tensorboard_callback])
    plt.plot(training_history.history['loss']); plt.show()
    simple_lstm_model.save_weights(os.path.join(args.model_dir, 'simple_lstm_model'))

    prediction_model = create_lstm_model(input_shape=data_gen.input_shape, output_shape=data_gen.output_shape,
                                         stateful=True, batch_sz=1, sequence_len=1)
    prediction_model.set_weights(simple_lstm_model.get_weights())
    # prediction_model.load_weights('simple_lstm_model')
    import ipdb; ipdb.set_trace()

    test_lstm_model(prediction_model, test_traj=trajectories[:1], args=args)


def train_mlp_model(data_gen, trajectories, args):
    simple_mlp_model = create_mlp_model(input_shape=data_gen.input_shape, output_shape=data_gen.output_shape)
    simple_mlp_model.compile('adam', 'mean_squared_error')

    log_dir = "logs/" + datetime.datetime.now().strftime("mlp_model_%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    training_history = simple_mlp_model.fit(data_gen.generate(args.batch_sz), epochs=args.epochs,
                                            steps_per_epoch=50 * len(trajectories) / args.batch_sz,
                                            validation_data=data_gen.generate(args.batch_sz), validation_steps=10,
                                            callbacks=[tensorboard_callback])
    plt.plot(training_history.history['loss']); plt.show()
    simple_mlp_model.save_weights(os.path.join(args.model_dir, 'simple_mlp_model'))

    X, Y = create_dataset_from_trajectories(trajectories[:1],
                                            data_gen_sampling_frequency=args.data_gen_sampling_frequency,
                                            measurement_sampling_frequency=args.measurement_sampling_frequency,
                                            future_horizons=args.future_horizons, history=args.history)
    print('X.shape: {}, \t Y.shape: {}'.format(X.shape, Y.shape))
    # plt.plot(*trajectories[0][:, :2].T)
    for i in range(0, len(X), 200):
        plt.plot(*trajectories[0][:, :2].T)
        plt.scatter(*X[i][:, :2].T, color='r')
        plt.scatter(*Y[i][:, :2].T, color='b')
        plt.show()

    Y_pred = simple_mlp_model.predict(X - X[:, 0:1, :]) + X[:, 0:1, :]
    print('Total Error: {}'.format(np.linalg.norm(Y_pred - Y, axis=2).mean(axis=0)))
    for j in range(Y.shape[1]):
        for i in range(0, Y.shape[0], 200):
            plt.plot([Y[i, j, 0], Y_pred[i, j, 0]], [Y[i, j, 1], Y_pred[i, j, 1]])
        plt.title('Error: {}'.format(np.mean(np.linalg.norm(Y[:, j, :] - Y_pred[:, j, :], axis=1))))
        plt.show()
    plt.plot(*trajectories[0][:, :2].T)
    import ipdb; ipdb.set_trace()


def load_model(model_file_path, input_shape, output_shape, is_lstm_model=False):
    if is_lstm_model:
        prediction_model = create_lstm_model(input_shape=input_shape, output_shape=output_shape,
                                             stateful=True, batch_sz=1, sequence_len=1)
        prediction_model.load_weights(model_file_path)
    else:
        prediction_model = create_mlp_model(input_shape=input_shape, output_shape=output_shape)
        prediction_model.load_weights(model_file_path)
    return prediction_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Motion Prediction Network')

    parser.add_argument('-n', '--num_data_points', type=int, default=2000)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_sz', type=int, default=1000)
    parser.add_argument('-md', '--model_dir', type=str, default='motion_model')
    parser.add_argument('-dl', '--distance_low', type=float, default=0.3, help='in meters')
    parser.add_argument('-dh', '--distance_high', type=float, default=0.7, help='in meters')
    parser.add_argument('-l', '--length', type=float, default=1.0, help='in meters')
    parser.add_argument('-s', '--speeds', nargs='+', type=float, default=(0.01, 0.03, 0.05), help='-f 0.01, 0.03, 0.05')
    parser.add_argument('-d', '--data_gen_sampling_frequency', type=float, default=240, help='in Hertz')
    parser.add_argument('-m', '--measurement_sampling_frequency', type=float, default=5, help='in Hertz')
    parser.add_argument('-f', '--future_horizons', nargs='+', type=float, default=(1., 2.), help='-f 1. 2.')
    parser.add_argument('-his', '--history', type=int, default=5)
    parser.add_argument('-i', '--is_sinusoid', action='store_true', default=False)
    parser.add_argument('-t', '--train_lstm', action='store_true', default=False)
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    trajectories = create_dataset(num_data_points=args.num_data_points,
                                  sampling_frequency=args.data_gen_sampling_frequency, is_sinusoid=args.is_sinusoid,
                                  distance_low=args.distance_low, distance_high=args.distance_high, length=args.length,
                                  speeds=args.speeds)

    data_gen = DataGenerator(trajectories, data_gen_sampling_frequency=args.data_gen_sampling_frequency,
                             measurement_sampling_frequency=args.measurement_sampling_frequency,
                             future_horizons=args.future_horizons, history=args.history)
    # sample = data_gen.generate(2).__next__()
    # sample = data_gen.generate_sequence(2).__next__()

    # import ipdb;    ipdb.set_trace()
    # plt.plot(*trajectories[0][:, :2].T)
    # for i in range(0, len(trajectories[0]), 200):
    #     plt.scatter(*X[i][:, :2].T, color='r')
    #     plt.scatter(*Y[i][:, :2].T, color='b')
    # plt.show()

    # import ipdb; ipdb.set_trace()

    if args.train_lstm:
        train_lstm_model(data_gen, trajectories, args)
    else:
        train_mlp_model(data_gen, trajectories, args)
