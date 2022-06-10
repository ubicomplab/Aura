import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import pickle
import seaborn as sns

CALIBRATION_FILE = r"D:\mag_track\calibration\helmholz_18_12_07_22_52_11.txt"

PER_CHANNEL_GAIN = True
CROSSTALK_MATRIX = False
CROSS_BIAS = False
BIAS = False
POLY_FIT = False
POLY_ORDER = 11

USE_INTERP_CORRECTION = False
USE_SIGN_CHANGE = False


def apply_calibration(data, per_channel_gain, crosstalk, cross_bias, bias, poly):
    # bias = [-.005, -.005, -.005]
    poly_func = np.poly1d(poly)
    data = np.sign(data) * poly_func(np.abs(data))
    # cross_bias = [-3.9e-5, -3.9e-5, -3.9e-5]
    # calibrated = np.sqrt(((data + bias) / per_channel_gain) ** 2 - cross_bias**2)
    noise, gain, bias = (0.1761193,  0.58924042, 0.09726018)
    bias = gain * noise - .0051
    # mag_data = np.sign(mag_data) * np.sqrt(((np.abs(mag_data) + 0.08888841 -.00059) / 1.53998297) ** 2 - 0.06149381 ** 2)
    calibrated = np.sqrt(((np.abs(data) + bias) / gain) ** 2 - noise**2)
    # calibrated = np.sign(data) * np.sqrt(data**2 + cross_bias)
    # calibrated = np.sign(data) * (np.abs(calibrated) + bias)
    calibrated = np.nan_to_num(calibrated)
    calibrated = np.matmul(calibrated, np.diag(per_channel_gain))
    # calibrated = np.sqrt(np.matmul(calibrated**2, crosstalk))

    calibrated = calibrated * np.sign(data)
    calibrated = np.matmul(calibrated, crosstalk)
    # for axis in range(3):
    #     negatives = np.sign(data[:,axis]) < 0
    #     calibrated[negatives, axis] = calibrated[negatives, axis] + bias[axis]

    return calibrated

def apply_calibration2(data, per_channel_gain, crosstalk, cross_bias, bias, poly):
    # bias = [-.005, -.005, -.005]
    poly_func = np.poly1d(poly)
    data = np.sign(data) * poly_func(np.abs(data))
    # cross_bias = [-3.9e-5, -3.9e-5, -3.9e-5]
    # calibrated = np.sqrt(((data + bias) / per_channel_gain) ** 2 - cross_bias**2)
    noise, gain, bias = (0.1761193,  0.58924042, 0.09726018)
    bias = gain * noise - .0051
    # mag_data = np.sign(mag_data) * np.sqrt(((np.abs(mag_data) + 0.08888841 -.00059) / 1.53998297) ** 2 - 0.06149381 ** 2)
    calibrated = data
    # calibrated = np.sign(data) * np.sqrt(data**2 + cross_bias)
    # calibrated = np.sign(data) * (np.abs(calibrated) + bias)
    calibrated = np.nan_to_num(calibrated)
    calibrated = np.matmul(calibrated, np.diag(per_channel_gain))
    # calibrated = np.sqrt(np.matmul(calibrated**2, crosstalk))

    calibrated = calibrated * np.sign(data)
    calibrated = np.matmul(calibrated, crosstalk)
    # for axis in range(3):
    #     negatives = np.sign(data[:,axis]) < 0
    #     calibrated[negatives, axis] = calibrated[negatives, axis] + bias[axis]

    return calibrated


def apply_calibration_x(data, x):
    per_channel_gain, crosstalk, cross_bias, bias, poly = decode_x(x)
    return apply_calibration(data, per_channel_gain, crosstalk, cross_bias, bias, poly)

def apply_calibration_x2(data, x):
    per_channel_gain, crosstalk, cross_bias, bias, poly = decode_x(x)
    return apply_calibration2(data, per_channel_gain, crosstalk, cross_bias, bias, poly)


def measure_error(calibrated, calibrated_samples):
    mag = np.linalg.norm(calibrated, axis=1)
    error = mag - 1
    error = error[~np.isnan(error)]
    error_mag = np.mean(error**2)*2

    if USE_SIGN_CHANGE:
        samples = [sample[:, dim] for sample, dim in calibrated_samples]
        samples = np.nan_to_num(samples)
        x = np.linspace(0, 1, len(samples[0]))
        rs = [scipy.stats.pearsonr(x, y)[0] for y in samples]
        rs = np.nan_to_num(rs)
        # print(matrix, bias, np.mean(error**2))
        error_sign = np.mean((np.array(rs)**2-1)**2)*5
    else:
        error_sign = 0

    return error_mag + error_sign



def filter_bad_data(data, use_abs=False):
    print(data.shape)

    if use_abs:
        transform = lambda x: np.abs(x)
    else:
        transform = lambda x: x

    filtered = scipy.signal.savgol_filter(transform(data), 15, 2, axis=0)
    error = np.sqrt(np.sum((transform(data)-filtered)**2, axis=1))
    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(error)
    # plt.show()
    return data[error < .007, :]


def encode_x(per_channel_gain, crosstalk, cross_bias, bias, poly):
    x = []
    if PER_CHANNEL_GAIN:
        x += per_channel_gain
    if CROSSTALK_MATRIX:
        x += list(crosstalk.flatten())
    if CROSS_BIAS:
        x += cross_bias
    if BIAS:
        x += bias
    if POLY_FIT:
        x += poly
    return np.array(x)


def decode_x(x):
    i = 0

    if PER_CHANNEL_GAIN:
        per_channel_gain = x[i:i+3]
        i += 3
    else:
        per_channel_gain = [1, 1, 1]

    if CROSSTALK_MATRIX:
        crosstalk = x[i:i+9].reshape(3, 3)
        i += 9
    else:
        crosstalk = np.eye(3)

    if CROSS_BIAS:
        cross_bias = x[i:i+3]
        i += 3
    else:
        cross_bias = [0, 0, 0]

    if BIAS:
        bias = x[i:i+3]
    else:
        bias = [0, 0, 0]

    if POLY_FIT:
        poly = x[i:i+POLY_ORDER]
    else:
        poly = [0] * (POLY_ORDER-2) + [1, 0]
    return per_channel_gain, crosstalk, cross_bias, bias, poly


def get_bounds():
    bounds = []
    if PER_CHANNEL_GAIN:
        bounds += [(.1,4)] * 3

    if CROSSTALK_MATRIX:
        for i in range(9):
            if i % 4 == 0:
                bounds += [(.999, 1.001)]
            else:
                bounds += [(-.5, 0)]

    if CROSS_BIAS:
        bounds += [(-.2, 0)] * 3
    if BIAS:
        bounds += [(-.1, .1)] * 3
    if POLY_FIT:
        bounds += [(-20, 20)] * (POLY_ORDER-1) + [(-7e-03, 7e-03)]

    return bounds


def find_zero_crossings(data):
    samples = []
    plt.figure()
    for (i, dim) in zip(*np.where(np.diff(np.sign(data), axis=0))):
        diff = data[i + 1, dim] - data[i, dim]
        if diff < 0.1:
            print(diff)
            sample = data[i-500:i+500, :]
            plt.plot(sample[:,dim])
            samples.append((sample, dim))
    plt.show()
    return samples


def fix_signs(data):
    # plt.plot(data)
    for (i, dim) in zip(*np.where(np.diff(np.sign(data), axis=0))):
        no_switch_diff = data[i + 1, :] - data[i, :]
        switch_diff = -data[i + 1, :] - data[i, :]
        if np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff):
            data[i+1:, :] *= -1

    # plt.figure()
    # plt.plot(data)
    # plt.show()
    return data


def main(calibration_file):
    print("Loading data...")
    data = np.loadtxt(calibration_file, delimiter=",")
    # plt.plot(data)
    # plt.show()
    data = filter_bad_data(data, use_abs=True)
    data = fix_signs(data)
    data = filter_bad_data(data)

    data = data[::100]

    if USE_INTERP_CORRECTION:
        func = pickle.load(open("interp.pkl", 'rb'))
        data = np.sign(data) * (np.abs(data) + func(np.abs(data)))
        plt.plot(data)
        plt.show()
    if USE_SIGN_CHANGE:
        samples = find_zero_crossings(data)
    else:
        samples = []
    # data = data[np.abs(data[:,0]) > .1]
    # plt.plot(data)
    # data = scipy.signal.savgol_filter(data, 31, 2, axis=0)

    def cost(x):
        calibrated = apply_calibration_x(data, x)
        calibrated_samples = [(apply_calibration_x(sample, x), dim) for sample, dim in samples]
        error = measure_error(calibrated, calibrated_samples)
        return error
    def cost2(x):
        calibrated = apply_calibration_x2(data, x)
        calibrated_samples = [(apply_calibration_x2(sample, x), dim) for sample, dim in samples]
        error = measure_error(calibrated, calibrated_samples)
        return error
    x0 = np.hstack((np.eye(3).flatten(), [0,0,0]))
    per_channel_gain = [1,1,1]
    crosstalk = np.eye(3)
    cross_bias = [0, 0, 0]
    bias = [0, 0, 0]
    poly = [0] * (POLY_ORDER-2) + [1, 0]
    x0 = encode_x(per_channel_gain, crosstalk, cross_bias, bias, poly)
    print("Running minimizer...")
    x_opt = scipy.optimize.minimize(cost, x0=x0, bounds=get_bounds())
    x_opt2 = scipy.optimize.minimize(cost2, x0=x0, bounds=get_bounds())
    print(x_opt)

    calibrated = apply_calibration_x(data, x_opt.x)
    calibrated2 = apply_calibration_x(data, x0)
    calibrated3 = apply_calibration_x2(data, x_opt2.x)

    # per_channel_gain = [1.73543888, 1.73543888, 1.73543888]
    # bias = [0, 0, 0]
    # crosstalk = np.eye(3)
    # crosstalk[0,1] = -5e-4
    # crosstalk[2,1] = -5e-4
    # cross_bias = [1, 1, 1]
    # x0 = encode_x(per_channel_gain, crosstalk, cross_bias, bias)
    # calibrated2 = apply_calibration_x(data, x0)
    measure_error(data, [(apply_calibration_x(sample, [0, 0, 0]), dim) for sample, dim in samples])
    plt.figure()
    plt.plot(data)
    plt.figure()
    plt.plot(calibrated)
    plt.figure()
    plt.plot(np.abs(calibrated))
    # plt.figure()
    # plt.plot(calibrated2)

    sns.set(context="paper", style="white", font="Lato")
    fig = plt.figure(figsize=(3.3, 3))
    ax = fig.add_subplot(111)

    t = np.linspace(0, len(data) / 40, len(data))
    ax.plot(t, np.linalg.norm(data, axis=1) / np.mean(np.linalg.norm(data, axis=1)), label="No calibration")
    ax.plot(t, np.linalg.norm(calibrated2, axis=1) / np.mean(np.linalg.norm(calibrated2, axis=1)), label="Noise Calibration")
    ax.plot(t, np.linalg.norm(calibrated3, axis=1), label="Gain Calibration")
    ax.plot(t, np.linalg.norm(calibrated, axis=1), label="Noise + Gain Calibration")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Field magnitude (normalized)")
    plt.legend()
    fig.subplots_adjust(bottom=0.20)
    plt.show()


if __name__ == "__main__":
    main(CALIBRATION_FILE)
