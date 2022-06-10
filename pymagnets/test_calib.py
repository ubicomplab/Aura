import pickle
from scipy.optimize import leastsq
import numpy as np
from matplotlib import pyplot as plt

def main():

    p = pickle.load(open('calib.pkl', 'rb'))

    i_amp, i_mean = fit(p['i'])
    q_amp, q_mean = fit(p['q'])

def fit(data):
    model = lambda x: x[0]*np.sin(np.linspace(0, 2*np.pi*x[1], N)+x[2]) + x[3]
    N = len(data)
    mean = np.mean(data)
    freq = 3 # cycles per N samples
    amp = 2*np.std(data)/(2**0.5)
    phase = 0
    data_first_guess = model([amp, freq, phase, mean])
    plt.plot(data, '.')
    plt.plot(data_first_guess)
    optimize_func = lambda x: model(x) - data
    amp, freq, phase, mean = leastsq(optimize_func, [amp, freq, phase, mean])[0]

    data_opt = model([amp, freq, phase, mean])
    print(amp)
    print(mean)
    plt.plot(data_opt)
    plt.show()
    return amp, mean


if __name__ == "__main__":
    main()