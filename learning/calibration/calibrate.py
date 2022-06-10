import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import pickle
from pyquaternion import Quaternion
from preprocessing.step1_extract_coordinates import extract_coordinates, extract_controller_coord_no_head
from utils import load_opti_data, load_buffer_data
from preprocessing.step2_resample_and_filter import resample_and_align, transform_rot, preprocess_calibration

CALIBRATION_KEY = "helmholtz2"

PER_CHANNEL_GAIN = True
CROSSTALK_MATRIX = False
CROSS_BIAS = False
BIAS = False
POLY_FIT = False
POLY_ORDER = 11
SENSOR_ROT = False

USE_INTERP_CORRECTION = True
USE_SIGN_CHANGE = False


def apply_calibration(data, rot_quat, per_channel_gain, crosstalk, cross_bias, bias, poly, rot, apply_rot=True):

    # bias = [-.005, -.005, -.005]
    poly_func = np.poly1d(poly)
    data = np.sign(data) * poly_func(np.abs(data))
    calibrated = np.sign(data) * (np.abs(data) + bias)
    # cross_bias = [-3.9e-5, -3.9e-5, -3.9e-5]
    calibrated = np.sqrt(calibrated**2 + cross_bias)
    calibrated = np.nan_to_num(calibrated)
    calibrated = np.matmul(calibrated, np.diag(per_channel_gain))
    # calibrated = np.sqrt(np.matmul(calibrated**2, crosstalk))

    calibrated = calibrated * np.sign(data)
    calibrated = np.matmul(calibrated, crosstalk)
    if apply_rot:
        rotated = [(q * rot).rotate(x) for q, x in zip(rot_quat, calibrated)]
    else:
        rotated = calibrated
    # for axis in range(3):
    #     negatives = np.sign(data[:,axis]) < 0
    #     calibrated[negatives, axis] = calibrated[negatives, axis] + bias[axis]

    return rotated


def apply_calibration_x(data, x, rot_quat, apply_rot=True):
    per_channel_gain, crosstalk, cross_bias, bias, poly, rot = decode_x(x)
    return apply_calibration(data, rot_quat, per_channel_gain, crosstalk, cross_bias, bias, poly, rot, apply_rot)


def measure_error(calibrated, calibrated_samples):
    mag = np.linalg.norm(calibrated, axis=1)
    error = mag - 1
    error = error[~np.isnan(error)]
    error_mag = np.mean(error**2)*2
    # dot_prod = np.dot(calibrated, [-1,0,0])
    # error = dot_prod - 1
    # error_mag = np.mean(error ** 2) * 2

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


def encode_x(per_channel_gain, crosstalk, cross_bias, bias, poly, rot):
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
    if SENSOR_ROT:
        x += rot
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

    if SENSOR_ROT:
        rot = x[i:i+4]
    else:
        rot = [1,0,0,0]

    return per_channel_gain, crosstalk, cross_bias, bias, poly, rot


def get_bounds():
    bounds = []
    if PER_CHANNEL_GAIN:
        bounds += [(.5,5)] * 3

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
    if SENSOR_ROT:
        bounds += [(-1, 1)] * 4

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


def filter_and_downsample(x, factor):
    # b, a = scipy.signal.butter(8, 40 / (3960/2), btype='lowpass')
    # x_filt = scipy.signal.filtfilt(b, a, x, axis=0)
    x_filt = x
    # plt.plot(x)
    # plt.plot(x_filt)
    # plt.show()
    # return scipy.signal.decimate(x_filt, factor)
    return x_filt[::factor]

def main(key):
    print("Loading data...")
    print("processing opti file")
    opti_data = load_opti_data(key)
    print("done loading opti file")
    print(opti_data)

    data = load_buffer_data(key)
    data = fix_signs(data)

    pos, rot = extract_controller_coord_no_head(opti_data)

    mag_data, pos_interp, rot_interp = resample_and_align(data, pos, rot, preprocess_calibration, transformer=transform_rot, mag_rate=3960, opti_rate=120)
    mag_filt = filter_and_downsample(mag_data, 50)
    pos_filt = filter_and_downsample(pos_interp, 50)
    rot_filt = filter_and_downsample(rot_interp, 50)

    print(data)
    data = mag_filt

    rot_quat = [Quaternion(x) for x in rot_filt]
    # plt.figure()
    # plt.plot(rotated)
    # plt.show()
    # data = filter_bad_data(data, use_abs=True)
    # data = fix_signs(data)
    # data = filter_bad_data(data)

    data = np.sign(data) * (np.abs(data) - np.array([.0028,.0028,.02]))
    if USE_INTERP_CORRECTION:
        func = pickle.load(open("interp.pkl", 'rb'))
        plt.figure()
        plt.plot(data)
        data = np.sign(data) * func(np.abs(data))
        plt.plot(data)
        plt.title("Effect of interpolation")
        # plt.show()

    # data = np.matmul(data, np.diag([2.49207791, 2.19823757, 2.92806613]))
    # data = np.matmul(data, np.array([[1,0,0],[0,0,-1],[0,1,0]]).T)

    if USE_SIGN_CHANGE:
        samples = find_zero_crossings(data)
    else:
        samples = []
    # data = data[np.abs(data[:,0]) > .1]
    # plt.plot(data)
    # data = scipy.signal.savgol_filter(data, 31, 2, axis=0)

    def cost(x):
        calibrated = apply_calibration_x(data, x, rot_quat)
        calibrated_samples = [(apply_calibration_x(sample, x, rot_quat), dim) for sample, dim in samples]
        error = measure_error(calibrated, calibrated_samples)
        print(x, error)
        return error

    x0 = np.hstack((np.eye(3).flatten(), [0,0,0]))
    per_channel_gain = [1,1,1]
    crosstalk = np.eye(3)
    cross_bias = [0, 0, 0]
    bias = [0, 0, 0]
    poly = [0] * (POLY_ORDER-2) + [1, 0]
    rot = [1, 0, 0, 0]
    x0 = encode_x(per_channel_gain, crosstalk, cross_bias, bias, poly, rot)
    print("Running minimizer...")
    x_opt = scipy.optimize.minimize(cost, x0=x0, bounds=get_bounds())
    print(x_opt)

    calibrated = apply_calibration_x(data, x_opt.x, rot_quat, apply_rot=False)
    calibrated_rot = apply_calibration_x(data, x_opt.x, rot_quat, apply_rot=True)

    # per_channel_gain = [1.73543888, 1.73543888, 1.73543888]
    # bias = [0, 0, 0]
    # crosstalk = np.eye(3)
    # crosstalk[0,1] = -5e-4
    # crosstalk[2,1] = -5e-4
    # cross_bias = [1, 1, 1]
    # x0 = encode_x(per_channel_gain, crosstalk, cross_bias, bias)
    # calibrated2 = apply_calibration_x(data, x0)
    measure_error(data, [(apply_calibration_x(sample, [0, 0, 0], rot_quat), dim) for sample, dim in samples])
    plt.figure()
    plt.title("Original data (after interpolation)")
    plt.plot(data)
    plt.figure()
    plt.title("Calibrated data")
    plt.plot(calibrated)
    plt.figure()
    plt.title("Abs Calibrated data")
    plt.plot(np.abs(calibrated))
    plt.figure()
    plt.title("Calibrated data after rotation")
    plt.plot(calibrated_rot)
    # plt.figure()
    # plt.plot(calibrated2)

    plt.figure()
    plt.title("Magnitudes")
    plt.plot(np.linalg.norm(data, axis=1))
    plt.plot(np.linalg.norm(calibrated, axis=1))
    plt.show()

if __name__ == "__main__":
    main(CALIBRATION_KEY)
