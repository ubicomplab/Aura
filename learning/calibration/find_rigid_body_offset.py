import os
import pickle

import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

from settings import DATA_ROOT

CALIBRATION_FILE = r'D:\mag_track\rigid_body_calibration\hand\markers_18_12_05_19_23_00.txt'
NUM_MARKERS = 5
MARKER_DISTANCE_THRESHOLD_MM = 130

HAND_MARKERS = np.array([[85.75, 0, 0],
                        # [0, 53.4, 0],
                        [-4, 53.2, 0],
                        [-84.25, -2.3, -16.49],
                        [85.75, -1.37, -55.80],
                        [-84.25, -2.3, -52.24]])

HEAD_MARKERS = np.array([[-95, 0, 0],
                        [95, 0, 0],
                        [-54, 95, -45],
                        [-26, 95, -85]])


def find_correspondences(vicon_markers, cad_markers):

    all_distances_vicon = np.zeros((NUM_MARKERS, NUM_MARKERS))
    all_distances_cad = np.zeros((NUM_MARKERS, NUM_MARKERS))
    for i in range(NUM_MARKERS):
        distances = np.linalg.norm(vicon_markers - vicon_markers[i,:], axis=1)
        all_distances_vicon[i, :] = distances

        distances = np.linalg.norm(cad_markers - cad_markers[i,:], axis=1)
        all_distances_cad[i, :] = distances

    vicon_markers_reordered = []
    for i in range(NUM_MARKERS):
        target_distances = sorted(all_distances_cad[i, :])
        errors = []
        for vicon_i in range(NUM_MARKERS):
            vicon_distance = np.sort(all_distances_vicon[vicon_i, :])
            error = np.linalg.norm(target_distances - vicon_distance)
            errors.append(error)
        vicon_markers_reordered.append(vicon_markers[np.argmin(errors), :])

    return np.array(vicon_markers_reordered)


def find_transform(vicon_markers, cad_markers):
    vicon_centroid = np.mean(vicon_markers, axis=0)
    cad_centroid = np.mean(cad_markers, axis=0)
    vicon_centered = vicon_markers - vicon_centroid
    cad_centered = cad_markers - cad_centroid

    H = np.matmul(vicon_centered.T, cad_centered)
    u, s, v = np.linalg.svd(H)
    r = np.matmul(v, u.T)
    t = np.matmul(-r, vicon_centroid) + cad_centroid

    cad_origin = np.array([0,0,0])
    vicon_cad_origin = -np.matmul(r.T, t).T
    print(vicon_cad_origin)

    # cad_trans = np.matmul(r, vicon_markers.T).T + t
    cad_trans = np.matmul(r.T, (cad_markers - t).T).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vicon_markers[:, 0], vicon_markers[:, 1], vicon_markers[:, 2],
               c=list(range(NUM_MARKERS)), alpha=1, marker='^')
    ax.scatter(cad_trans[:, 0], cad_trans[:, 1], cad_trans[:, 2], c=list(range(NUM_MARKERS)), alpha=1,
               marker='.')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("After transformation")
    # plt.show()

    return r, vicon_cad_origin


def find_markers_for_body(markers, pos, q, estimated_markers):
    # compute distances to filter by frames with N nearby markers
    offsets = markers - pos
    num_markers_nearby = np.sum(np.linalg.norm(offsets, axis=2) < MARKER_DISTANCE_THRESHOLD_MM, axis=0)
    good_frames = num_markers_nearby == NUM_MARKERS

    # filter down to good frames
    markers = markers[:, good_frames, :]
    pos = pos[good_frames, :]
    qs = [Quaternion(_q) for _q in q[good_frames, :]]
    offsets = markers - pos

    # rotate offsets by body rotation
    distances = np.linalg.norm(offsets, axis=2)
    offsets_body_space = np.array([[q.conjugate.rotate(offset) for q, offset in zip(qs, marker_offsets)] for marker_offsets in offsets])

    # grab all markers that are close by
    all_markers = np.zeros((0, 3))
    for frame_idx in range(markers.shape[1]):
        near_markers_body_space = offsets_body_space[distances[:, frame_idx] < MARKER_DISTANCE_THRESHOLD_MM, frame_idx, :]
        all_markers = np.vstack((near_markers_body_space, all_markers))


    # cluster markers
    clustering = DBSCAN(eps=3, min_samples=20).fit(all_markers)
    sizes = []
    unique_labels = np.unique(clustering.labels_)
    for label in unique_labels:
        count = np.sum(clustering.labels_ == label)
        print(f"Label {label}: {count}")
        sizes.append(count)
    top_indices = np.array(sizes).argsort()[-NUM_MARKERS:][::-1]
    top_labels = unique_labels[top_indices]

    clustering.labels_[~np.isin(clustering.labels_, top_labels)] = -1

    marker_locations = np.zeros((NUM_MARKERS, 3))
    for marker_idx, marker_id in enumerate(top_labels):
        points = all_markers[clustering.labels_ == marker_id, :]
        marker_locations[marker_idx, :] = np.median(points, axis=0)

    ordered_marker_locations = find_correspondences(marker_locations, estimated_markers)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_markers[:,0], all_markers[:,1], all_markers[:,2], c=clustering.labels_, alpha=.1)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Clustering results")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ordered_marker_locations[:,0], ordered_marker_locations[:,1], ordered_marker_locations[:,2], c=list(range(NUM_MARKERS)), alpha=1, marker='^')
    ax.scatter(estimated_markers[:,0], estimated_markers[:,1], estimated_markers[:,2], c=list(range(NUM_MARKERS)), alpha=1, marker='.')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Correspondences")
    # fig = plt.figure()
    # plt.plot(all_markers[:,0], '.')
    # fig = plt.figure()
    # plt.plot(all_markers[:,1], '.')
    # fig = plt.figure()
    # plt.plot(all_markers[:,2], '.')
    # plt.show()
    print(offsets_body_space.shape)
    return ordered_marker_locations


def adjust_body(body_p, body_q, r, t):
    body_q = Quaternion(body_q)
    adjust_p = body_p + body_q.rotate(t)
    rq = Quaternion(matrix=r)
    adjust_q = body_q * rq.conjugate
    return adjust_p, adjust_q.elements


CHECK_FRAME = [3360, 4704]


def plot_q(ax, origin, q, size=10):
    q = Quaternion(q)
    x = q.rotate([size,0,0]) + origin
    y = q.rotate([0,size,0]) + origin
    z = q.rotate([0,0,size]) + origin
    ax.plot([origin[0], x[0]], [origin[1], x[1]], [origin[2], x[2]], color='r')
    ax.plot([origin[0], y[0]], [origin[1], y[1]], [origin[2], y[2]], color='g')
    ax.plot([origin[0], z[0]], [origin[1], z[1]], [origin[2], z[2]], color='b')


def main():
    data = np.loadtxt(CALIBRATION_FILE, delimiter=',')
    head_pos = data[:, 0:3]
    head_q = data[:, 3:7]
    hand_pos = data[:, 7:10]
    hand_q = data[:, 10:14]
    markers = data[:, 14:]
    markers = np.array(np.split(markers, 10, axis=1))

    print(markers[:,:,0].shape)
    # plt.plot(markers[:,:,0].T, '.')
    # plt.show()

    avg_markers = find_markers_for_body(markers, hand_pos, hand_q, HAND_MARKERS)
    hand_r, hand_t = find_transform(avg_markers, HAND_MARKERS)
    avg_markers = find_markers_for_body(markers, head_pos, head_q, HEAD_MARKERS)
    head_r, head_t = find_transform(avg_markers, HEAD_MARKERS)

    for frame in CHECK_FRAME:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(markers[:, frame, 0], markers[:, frame, 1], markers[:, frame, 2], alpha=1, marker='^')
        adjusted_p, adjusted_q = adjust_body(hand_pos[frame,:], hand_q[frame,:], hand_r, hand_t)
        ax.scatter(adjusted_p[0], adjusted_p[1], adjusted_p[2], alpha=1, marker='.', color='r')
        ax.scatter(hand_pos[frame,0], hand_pos[frame,1], hand_pos[frame,2], alpha=1, marker='.')

        ax.set_xlim(adjusted_p[0]-100, adjusted_p[0]+100)
        ax.set_ylim(adjusted_p[1]-100, adjusted_p[1]+100)
        ax.set_zlim(adjusted_p[2]-100, adjusted_p[2]+100)

        plot_q(ax, hand_pos[frame, :], hand_q[frame, :])
        plot_q(ax, adjusted_p, adjusted_q)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(markers[:, frame, 0], markers[:, frame, 1], markers[:, frame, 2], alpha=1, marker='^')
        adjusted_p, adjusted_q = adjust_body(head_pos[frame, :], head_q[frame, :], head_r, head_t)
        ax.scatter(adjusted_p[0], adjusted_p[1], adjusted_p[2], alpha=1, marker='.', color='r')
        ax.scatter(head_pos[frame, 0], head_pos[frame, 1], head_pos[frame, 2], alpha=1, marker='.')

        ax.set_xlim(adjusted_p[0] - 100, adjusted_p[0] + 100)
        ax.set_ylim(adjusted_p[1] - 100, adjusted_p[1] + 100)
        ax.set_zlim(adjusted_p[2] - 100, adjusted_p[2] + 100)

        plot_q(ax, head_pos[frame, :], head_q[frame, :])
        plot_q(ax, adjusted_p, adjusted_q)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    data = {'head_r': head_r, 'head_t': head_t, 'hand_r': hand_r, 'hand_t': hand_t}
    base_name = os.path.splitext(os.path.basename(CALIBRATION_FILE))[0]
    pickle.dump(data, open(os.path.join(DATA_ROOT, 'rigid_body_calibration', f'calib_{base_name}_v2.pkl'), 'wb'))
    plt.show()


if __name__ == "__main__":
    main()