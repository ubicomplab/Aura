import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import scipy.interpolate
import pickle
import seaborn as sns

# CALIBRATION_FILE = "calibrations/gain_calibration_18_11_15_18_07_15.txt"
from settings import DATA_ROOT

CALIBRATION_FILES = ["gain_calibration_18_11_15_18_07_15.txt", "gain_calibration_18_12_07_16_10_44.txt", "gain_calibration_18_12_07_16_17_16.txt", "gain_calibration_18_12_07_16_43_20.txt"]




def main(calibration_files):
    print("Loading data...")
    funcs = []
    all_data = []
    for idx, file in enumerate(calibration_files):
        print(file)
        data = np.loadtxt(os.path.join(DATA_ROOT, 'calibration', file), delimiter=",")
        voltage = data[:,0]
        c1 = data[:,1]
        print(max(c1))
        stop_idx = np.where(c1>0.8)[0][0]

        # interpolate to find a reference voltage
        func = scipy.interpolate.interp1d(c1, voltage, fill_value='extrapolate')
        ref_voltage = func(0.5)
        voltage = voltage / ref_voltage
        print(f"Ref voltage: {ref_voltage}")

        # print(c1, voltage)
        #
        # plt.figure()
        # plt.plot(c1, voltage)
        # plt.figure()
        # plt.plot(voltage[1:], np.diff(c1))


        # plt.figure()
        # error = voltage - c1
        # plt.plot(voltage, error)
        # plt.plot(c1, error)

        voltage_low = voltage[:stop_idx]
        c1_low = c1[:stop_idx]


        all_data += [(c1_low, voltage_low)]

        func = scipy.interpolate.interp1d(c1, voltage, fill_value='extrapolate')
        pickle.dump(func, open("interp.pkl", 'wb'))
        funcs.append(func)

        linear_func = scipy.interpolate.interp1d([voltage_low[0], voltage_low[-1]], [c1_low[0], c1_low[-1]], fill_value='extrapolate')
        func = scipy.interpolate.interp1d(voltage, c1, fill_value='extrapolate')
        def transform(x):
            noise = x[0]
            gain = x[1]
            bias = x[2]
            return np.sqrt(voltage_low**2+noise**2)*gain-bias
            # return np.sqrt(voltage_low**2+noise**2)*gain

        def cost(x):
            c1_hat = transform(x)
            error = c1_low - c1_hat
            return np.mean(error**2)
        opt = scipy.optimize.minimize(cost, x0=np.array([0,1,0]), bounds=[(1e-6, 1), (1e-6, 3), (-1, 1)])
        # opt = scipy.optimize.minimize(cost, x0=np.array([0,1]), bounds=[(1e-6, 1), (1e-6, 3)])
        print(opt.x)
        c1_hat = transform(opt.x)

        sns.set(context="paper", style="white", font="Lato")
        fig = plt.figure(figsize=(3.3, 2))
        ax = fig.add_subplot(111)
        t = np.linspace(0, (20700 - 18000) / 90, 20700 - 18000)
        ax.scatter(voltage, c1, label="Observed", marker='.', color="black")
        ax.plot(voltage_low, c1_hat, label="Noise model prediction", lw=1)
        ax.plot(voltage_low, linear_func(voltage_low), label="Linear")
        ax.set_xlabel("Field strength (normalized)")
        ax.set_ylabel("Voltage")
        ax.set_ylim((0, .4))
        ax.set_xlim((0, .6))
        plt.legend()
        fig.subplots_adjust(bottom=0.20)


        plt.show()

        def transform2(x):
            noise = x[0]
            gain = x[1]
            bias = x[2]
            # return np.sqrt(voltage_low**2+x[0]**2)*x[1]-x[2]
            return np.nan_to_num(np.sqrt(((np.abs(c1_low) + bias) / gain)**2 - noise**2))
            # return np.nan_to_num(np.sqrt(((np.abs(c1_low)) / gain)**2 - noise**2))

        def cost2(x):
            voltage_hat = transform2(x)
            error = voltage_low - voltage_hat
            # print(np.mean(error**2))
            return np.mean(error**2)

        # opt = scipy.optimize.minimize(cost2, x0=np.array([0.05,1.5,0.08]))
        opt = scipy.optimize.minimize(cost2, x0=opt.x, bounds=[(1e-6, 1), (1e-6, 3), (-1, 1)])
        # opt = scipy.optimize.minimize(cost2, x0=opt.x, bounds=[(1e-6, 1), (1e-6, 3)])
        print(opt.x)
        voltage_hat = transform2(opt.x)
        # plt.figure()
        # plt.plot(c1_low, voltage_low)
        # plt.plot(c1_low, voltage_hat)
        # plt.show()


        # poly_coeffs = np.polyfit(c1[:stop_idx], voltage[:stop_idx], 5)
        # print(poly_coeffs)
        # poly_func = np.poly1d(poly_coeffs)

        # plt.figure()
        # plt.plot(c1, voltage)
        # plt.plot(c1[:stop_idx], func(c1[:stop_idx]))
        # plt.show()

    def transform(x, voltage_low):
        noise = x[0]
        gain = x[1]
        bias = x[2]
        return np.sqrt((voltage_low) ** 2 + noise ** 2) * gain - bias

    def cost(x):
        all_error = 0
        for idx, (c1_low, voltage_low) in enumerate(all_data):
            c1_hat = transform(x, voltage_low)
            error = (c1_low - c1_hat) / c1_low[-1]
            all_error += np.mean(error ** 2)
        return all_error

    opt = scipy.optimize.minimize(cost, x0=np.array([0, 1, 0]), bounds=[(1e-6, 1), (1e-6, 3), (-1, 1)])
    print(opt.x)
    plt.figure()
    for idx, (c1_low, voltage_low) in enumerate(all_data):
        c1_hat = transform(opt.x, voltage_low)
        plt.plot(c1_low, c1_hat)



    # plt.figure()
    # sensor_vals = np.linspace(0,.5,1000)
    # scales = [1, 1]
    # for func, scale in zip(funcs, scales):
    #     scale = 1/ func(sensor_vals[-1])
    #     plt.plot(sensor_vals, scale * func(sensor_vals))

    plt.show()


if __name__ == "__main__":
    main(CALIBRATION_FILES)
