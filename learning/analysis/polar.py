from make_dataset import MAG
from utils import load_resampled_data, MAG_RAW_NAMES, OPTI_NAMES
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import scipy.optimize
from pyquaternion import Quaternion

TRIAL = 't1_march10'
FORWARD = [0,0,1]
TX1_POS = [0,0,0]
TX2_POS = [.2,0,-.05]
TX3_POS = [-.2,0,-.05]
Q_TX1 = Quaternion(axis=(1.0,0.0,0.0), degrees=0)
Q_TX2 = Q_TX1 * Quaternion(axis=(0.0,1.0,0.0), degrees=30)
Q_TX3 = Q_TX1 * Quaternion(axis=(0.0,1.0,0.0), degrees=-30)


def load_sim_data():
    df = pd.read_csv(r"C:\Users\emwhit\projects\mag-fingers\magfield_simulations\dipole\sim_cross_dipole.csv", names=MAG_RAW_NAMES + OPTI_NAMES)
    w = df.qx.values.copy()  # switch because matlab saves it differently
    x = df.qw.values.copy()
    assert(False, "This is wrong")
    df.qw = w
    df.qx = x
    return df


def coil_sim(rx_pos, tx_pos, q_tx):
    pos = rx_pos - tx_pos
    r = np.linalg.norm(pos, axis=1)

    pos_in_frame = np.matmul(q_tx.rotation_matrix, pos.T).T
    x = pos_in_frame[:,0]
    y = pos_in_frame[:,1]
    z = pos_in_frame[:,2]
    B = np.stack((3*x*z/r**5, 3*y*z/r**5, (3*z**2-r**2)/r**5))
    return np.matmul(q_tx.conjugate.rotation_matrix, B).T


def forward(df, rx_gain, tx_gain, tx_offsets, k):
    # rx_gain[0,0] = 1
    # tx_gain[0,0] = 1
    rx_pos = np.vstack((df.x, df.y, df.z)).T

    h1_hat = coil_sim(rx_pos, TX1_POS + tx_offsets[0, :], Q_TX1)
    h2_hat = coil_sim(rx_pos, TX2_POS + tx_offsets[1, :], Q_TX2)
    h3_hat = coil_sim(rx_pos, TX3_POS + tx_offsets[2, :], Q_TX3)

    h_hat = np.stack((h1_hat, h2_hat, h3_hat), axis=-1)
    h_hat_tx = np.matmul(h_hat, tx_gain)
    h_norm_hat = np.linalg.norm(h_hat_tx, axis=1) * k

    # m_0, m_1, m_2 are all from tx = 0
    mag = df[MAG_RAW_NAMES].values.reshape((-1, 3, 3))
    mag = mag[:,:,[0,1,2]]

    # mag = np.roll(mag, -3, axis=1)
    # mag = np.matmul(mag, scipy.linalg.block_diag(rx_gain,rx_gain,rx_gain))
    mag = np.matmul(mag, rx_gain)
    h = np.linalg.norm(mag, axis=2)

    return h, h_norm_hat


def cost(x, df, plot=False):
    rx_gain, tx_gain, tx_offsets, k = decode(x)
    H, H_hat = forward(df, rx_gain, tx_gain, tx_offsets, k)
    H = H[:,0]
    H_hat = H_hat[:,0]
    if plot:
        plt.figure()
        plt.scatter(H_hat, H, marker='.')
        plt.show()

    err = np.mean((H - H_hat)**2)
    print(np.sqrt(err))
    return err


def decode(x0):
    # rx_gain = np.hstack(([1], x0[0:8])).reshape((3, 3))
    # tx_gain = np.hstack(([1], x0[8:16])).reshape((3, 3))
    rx_gain = x0[0:9].reshape((3, 3))
    # rx_gain = np.eye(3)
    tx_gain = x0[9:18].reshape((3, 3))
    tx_offsets = np.zeros((3,3))#x0[16:25].reshape((3, 3))
    # rot = x0[27]
    k = 1 # x0[9]
    return rx_gain, tx_gain, tx_offsets, k

# # gain is c1, c2, c3, c12, c13, c23
# def make_gain_matrix(g):
#     g1 = g[0]
#     g2 = g[1]
#     g3 = g[2]
#     g12 = g[3]
#     g13 = g[4]
#     g23 = g[5]
#     gain = np.array([[g1, g12, g13], [g12, g2, g23], [g13, g23, g3]])
#     return gain.reshape((-1))


def encode(rx_gain, tx_gain, tx_offsets, k):
    rx_gain = rx_gain.reshape((-1))
    tx_gain = tx_gain.reshape((-1))
    return np.hstack((rx_gain, tx_gain)) # tx_offsets.reshape((-1))


def polar(trial):
    df = load_resampled_data(trial, variant="")
    df = df[np.all(df[MAG_RAW_NAMES] < 17000, axis=1)]
    # df = load_sim_data()
    # rx_gain = np.eye(3)
    # tx_gain = []
    rx_gain = np.eye(3)
    tx_gain_matlab = [0.7878,    0.7428,    0.6994]
    tx_gain = np.zeros((3,3))
    for i in range(3):
        tx_gain[i,i] = tx_gain_matlab[i]
        tx_gain[(i+1)%3, i] = (1-tx_gain_matlab[i])/2
        tx_gain[(i+2)%3, i] = (1-tx_gain_matlab[i])/2
    # tx_gain = np.array([[.8678, 0.0661, 0.0661], [0.1185, 0.7631, 0.1185], [0.0483, 0.0483, 0.9035]]).T
    tx_offsets = np.zeros((3,3))
    x0 = encode(rx_gain, tx_gain, tx_offsets, 1000)
    # x0 = np.array([2.57866922e+01,-1.34297003e+01,1.69473648e+01,1.14322951e+01,-2.07536307e+01,1.86563890e+00,-1.71389779e+00,2.39605505e+00,9.97710722e+00,-2.86032143e-02,0.00000000e+00,0.00000000e+00,5.95973484e-03,1.00000000e+00,0.00000000e+00,6.13239198e-03,0.00000000e+00,1.00000000e+00,1.03974649e+03])
    res = scipy.optimize.minimize(cost, x0, args=df, options={'maxiter': 10, 'disp': True})
    print(res.x)
    print(cost(res.x, df, plot=True))
    print(cost(x0, df, plot=True))


if __name__ == "__main__":
    polar(TRIAL)