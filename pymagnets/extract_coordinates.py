import numpy as np



def extract_coordinates_vicon(data):

    # data = handle_missing_frames(data)

    # data = adjust_rigid_bodies(data)

    pos_vicon, rot_vicon, is_valid = extract_controller_coord(data)

# def fix_frame(vicon_data, start_frame):
#     start = vicon_data.index[start_frame]
#     end = vicon_data.index[start_frame + 1]
#     missing_frames = np.linspace(start + 1, end - 1, end - start - 1).astype('int')
#     for missing_frame in missing_frames:
#         vicon_data.loc[missing_frame] = None
#     return vicon_data
#
#
# def handle_missing_frames(vicon_data):
#     # print("Interpolating missing vicon frames")
#
#     # first let's deal with frame drops (probably from dropped UDP packets)
#     vicon_data = vicon_data.set_index('frame')
#     assert (np.sum(
#         np.diff(vicon_data.index.values) == 0) == 0)  # this won't work if we have duplicated frame numbers
#     frames_to_fix, = np.where(np.diff(vicon_data.index.values) != 1)
#     for frame in frames_to_fix:
#         vicon_data = fix_frame(vicon_data, frame)
#     vicon_data = vicon_data.sort_index()
#     vicon_data.interpolate(inplace=True)
#     print(f"Interpolated {len(frames_to_fix)} missing frames")
#
#     # next let's deal with duplicated data due to out of bounds hands bodies
#     frames_to_fix = vicon_data.index != vicon_data.hand_frame  # , = np.where(np.diff(vicon_data.hand_frame.values) == 0)
#     vicon_data.loc[frames_to_fix, ['hand_frame'] + HAND_OPTI] = None
#     vicon_data.interpolate(inplace=True)
#     print(f"Interpolated {np.sum(frames_to_fix)} missing controller frames")
#
#     # finally, confirm no out of bounds head
#     assert (np.sum(vicon_data.index != vicon_data.head_frame) == 0)
#     assert (np.sum(vicon_data.index != vicon_data.hand_frame) == 0)
#
#     return vicon_data

#
# def adjust_bodies(body_p, body_q, r, t):
#
#     all_adjust_p = []
#     all_adjust_q = []
#     print("Adjusting rigid bodies")
#     for i in range(len(body_p)):
#         if i % 5000 == 0:
#             progress(i / len(body_p))
#         adjust_p, adjust_q = adjust_body(body_p[i, :], body_q[i, :], r, t)
#         all_adjust_p.append(adjust_p)
#         all_adjust_q.append(adjust_q)
#     all_adjust_p = np.array(all_adjust_p)
#     all_adjust_q = np.array(all_adjust_q)
#     print(all_adjust_p.shape)
#
#     return all_adjust_p, all_adjust_q
#
# def adjust_rigid_bodies(vicon_data):
#     pickle_data = pickle.load(open(os.path.join(DATA_ROOT, 'rigid_body_calibration', RIGID_BODY_CALIBRATION), 'rb'))
#     print(pickle_data)
#     hand_p, hand_q = adjust_bodies(vicon_data[HAND_OPTI_POS].values, vicon_data[HAND_OPTI_ROT].values,
#                                    pickle_data['hand_r'], pickle_data['hand_t'])
#     head_p, head_q = adjust_bodies(vicon_data[HEAD_OPTI_POS].values, vicon_data[HEAD_OPTI_ROT].values,
#                                    pickle_data['head_r'], pickle_data['head_t'])
#     vicon_data[HAND_OPTI_POS] = hand_p
#     vicon_data[HAND_OPTI_ROT] = hand_q
#     vicon_data[HEAD_OPTI_POS] = head_p
#     vicon_data[HEAD_OPTI_ROT] = head_q
#     return vicon_data
#
# def find_bounds(data, s):
#     if CHOOSE_STARTSTOP:
#         print("Enter start and stop indices")
#         plt.figure()
#         plt.plot(np.array(data[HEAD_OPTI]))
#         plt.show()
#         s[S.DATA_START] = int(input("start:"))
#         s[S.DATA_END] = int(input("end:"))
#     else:
#         print("using entire region")
#         s[S.DATA_START] = 0
#         s[S.DATA_END] = len(data[HEAD_OPTI])
#     # save_extra_data(TRIAL, s)
#     return s

def extract_controller_coord(data):
    # hand_rel_pos_world = data.apply(lambda x: [x.hand_x - x.head_x, x.hand_y - x.head_y, x.hand_z - x.head_z], axis=1)
    hand_rel_pos_world = data[HAND_OPTI_POS].values - data[HEAD_OPTI_POS].values
    print("Max distance (m):", max(np.linalg.norm(hand_rel_pos_world, axis=1)[1:]))

    # TODO: get rid of first nan
    all_hand_rel_pos_head = []
    all_hand_q_hand_to_head = []
    for i in range(len(data)):
        if i % 5000 == 0:
            progress(i / len(data))
        x = data.iloc[i]
        head_q_loc_to_world = Quaternion(a=x.head_qw, b=x.head_qx, c=x.head_qy, d=x.head_qz)
        hand_q_loc_to_world = Quaternion(a=x.mag_controller_v9_qw, b=x.mag_controller_v9_qx,
                                         c=x.mag_controller_v9_qy, d=x.mag_controller_v9_qz)
        head_q_world_to_loc = head_q_loc_to_world.conjugate

        hand_rel_pos_head = head_q_world_to_loc.rotate(hand_rel_pos_world[i, :])
        hand_q_hand_to_head = hand_q_loc_to_world * head_q_world_to_loc

        all_hand_rel_pos_head.append(hand_rel_pos_head)
        all_hand_q_hand_to_head.append(hand_q_hand_to_head.elements)  # w, x, y, z

    print("Max distance (m):", max([np.linalg.norm(x) for x in all_hand_rel_pos_head[1:]]))
    hand_tracking_lost = np.all(data[HAND_OPTI][1:].values == data[HAND_OPTI][0:-1].values, axis=1)
    head_tracking_lost = np.all(data[HEAD_OPTI][1:].values == data[HEAD_OPTI][0:-1].values, axis=1)
    tracking_lost = np.logical_or(hand_tracking_lost, head_tracking_lost)
    is_valid = np.insert(True, 0, np.logical_not(tracking_lost))
    print(sum(is_valid) / len(data))

    return np.array(all_hand_rel_pos_head), np.array(all_hand_q_hand_to_head), is_valid

def extract_controller_coord_no_head(data):
    hand_pos = data[HAND_OPTI_POS].values
    hand_pos_centered = hand_pos - np.mean(hand_pos, axis=0)

    hand_rot = data[[HAND_OPTI_ROT[x] for x in [3, 0, 1,
                                                2]]].values  # rearranging to convert from motive order (xyzw) to pyquaternion order (wxyz)

    return hand_pos_centered, hand_rot

