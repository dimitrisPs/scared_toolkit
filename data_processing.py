import cv2
import numpy as np
from pathlib import Path


ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def calib_to_ocv(calib_path):

    # resolve new resulting calib file path
    txt_file_path = Path(calib_path)
    file_dir = txt_file_path.parent
    file_name = txt_file_path.stem
    save_path = file_dir/(file_name+'.yaml')

    # open calib file provided
    with open(calib_path, "r") as calib:
        # read calib values
        params_l, params_r = calib_parser(calib)
        rot_rodrig = cv2.Rodrigues(params_r['R'])[0].astype('float64')
        T = params_r['T'].astype('float64').transpose()
        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(params_l['M'],
                                                    params_l['D'],
                                                    params_r['M'],
                                                    params_r['D'],
                                                    (1280, 1024),
                                                    rot_rodrig,
                                                    T,
                                                    alpha=0,
                                                    flags=0)

        # create and store values using opencv format
        fs = cv2.FileStorage(str(save_path), cv2.FileStorage_WRITE)
        fs.write(name='R', val=params_r['R'])
        fs.write(name='T', val=params_r['T'])
        fs.write(name='M1', val=params_l['M'])
        fs.write(name='D1', val=params_l['D'])
        fs.write(name='M2', val=params_r['M'])
        fs.write(name='D2', val=params_r['D'])
        fs.write(name='R1', val=RL)
        fs.write(name='R2', val=RR)
        fs.write(name='P1', val=PL)
        fs.write(name='P2', val=PR)
        fs.write(name='Q', val=Q)
        fs.release()
        return True


def process_sample(left_img, right_img, scene_points, calib_data):

    rot_rodrig, _ = cv2.Rodrigues(calib_data['R'])
    T = calib_data['T'].astype('float64')

    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        calib_data['M1'], calib_data['D1'], calib_data['M2'], calib_data['D2'], (1280, 1024), rot_rodrig.astype('float64'), T.transpose())

    map1l, map2l = cv2.initUndistortRectifyMap(
        calib_data['M1'], calib_data['D1'], RL, PL, (1280, 1024), cv2.CV_32FC1)
    map1r, map2r = cv2.initUndistortRectifyMap(
        calib_data['M2'], calib_data['D2'], RR, PR, (1280, 1024), cv2.CV_32FC1)

    rect_left = cv2.remap(left_img, map1l, map2l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map1r, map2r, cv2.INTER_LINEAR)

    left_3d_points = scene_points.reshape(-1, 3)
    left_rect_img_pts, _ = cv2.projectPoints(
        left_3d_points, RL, np.zeros((3, 1)), PL[:, :-1], np.zeros((4, 1)))
    right_rect_img_pts, _ = cv2.projectPoints(
        left_3d_points, RR, PR[:, -1]/PR[0, 0], PR[:, :-1], np.zeros((4, 1)))

    left_rect_img_pts = left_rect_img_pts.reshape(1024, 1280, -1)
    right_rect_img_pts = right_rect_img_pts.reshape(1024, 1280, -1)

    left_r_map, right_r_map, i = create_lr_map(
        left_rect_img_pts, right_rect_img_pts)
    disparity_left = create_disp(left_r_map)
    disparity_right = create_disp(right_r_map)

    return rect_left, rect_right, disparity_left, disparity_right, Q


def rectify_image(src, _size, calib):
    map1l, map2l = cv2.initUndistortRectifyMap(calib['M1'],
                                               calib['D1'],
                                               calib['R1'],
                                               calib['P1'],
                                               _size,
                                               cv2.CV_32FC1)
    rectified = cv2.remap(src, map1l, map2l, cv2.INTER_LINEAR)
    return rectified


def read_calib_file(calib_filepath):
    calib_data = cv2.FileStorage(calib_filepath, cv2.FILE_STORAGE_READ)
    calib_keys = ['R', 'T', 'K1', 'D1', 'K2',
                  'D2', 'R1', 'P1', 'R2', 'P2', 'Q']
    calib = {}
    for key in calib_keys:
        calib[key] = calib_data.getNode(key).mat()
    return calib


def calib_parser(calib_data):
    data = calib_data.readlines()
    # first camera parameters idx 3
    param_left = camera_param_parser(data[3])

    param_right = camera_param_parser(data[5])

    return param_left, param_right


def values_to_array(values, out_size=(-1), _dtype=np.float32):
    output = [float(elem) for elem in values]
    output = np.asarray(output, dtype=_dtype)
    output = output.reshape(out_size)
    return output


def camera_param_parser(camera_calib_data):
    values = camera_calib_data.split()
    size = (int(float(values[0])), int(float(values[1])))
    camera_mat = values_to_array(values[2:11], (3, 3))
    dist_coeffs = values_to_array(values[11:19], (1, 8))
    R = values_to_array(values[19:28], (3, 3))
    T = values_to_array(values[28:31], (1, 3))
    return {'size': size, 'M': camera_mat, 'D': dist_coeffs, 'R': R, 'T': T}


def create_xyz(scene_points, file_path):
    with open(file_path, 'w') as f:
        for entry in scene_points:
            entry = ' '.join([str(elem)for elem in entry])
            f.write(entry+'\n')


def create_ply(scene_points, left_img, path):
    with open(path, 'w') as pcd:
        pcd.write(ply_header.format(len(scene_points)))
        for entry in scene_points:
            print(entry)
            print(entry[0], entry[1])
            bgr = left_img[entry[0], entry[1]]
            # print(entry[0], entry[1])
            xyz = entry[2:]
            # print(xyz)
            pcd_entry = [*xyz, *bgr[::-1]]
            # print(pcd_entry)
            pcd_entry = ' '.join([str(elem)for elem in pcd_entry])
            pcd.write(pcd_entry+'\n')


if __name__ == '__main__':
    calib_to_ocv(
        '/home/dimitrisps/TMI_dataset/Stereo_SD_d_all/Stereo_SD_d_all_1/Stereo_SD_d_all_1_Calibration.txt')
    calib = read_calib_file(
        '/home/dimitrisps/TMI_dataset/Stereo_SD_d_all/Stereo_SD_d_all_1/Stereo_SD_d_all_1_Calibration.yaml')

    left_o = cv2.imread(
        '../data/left.bmp')
    right_o = cv2.imread(
        '../data/right.bmp')

    left_und_img = cv2.imread(
        '../data/left_und.bmp')
    left_o = cv2.imread(
        '../data/left.bmp')
    right = cv2.imread(
        '/home/dimitrisps/TMI_dataset/Stereo_SD_d_all/Stereo_SD_d_all_1/Stereo_SD_d_all_1_IMG_REC_right.bmp')

    mask = cv2.imread(
        '../data/mask.png')
    

    d = cv2.imread('./r.png', cv2.IMREAD_GRAYSCALE)
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    RT2 = np.hstack((calib['R'], calib['T'].transpose()))
    P1 = calib['M1'].dot(RT1)
    P2 = calib['M2'].dot(RT2)

    mask_und = 255-mask

    mask_und = cv2.undistort(mask_und, calib['M1'], calib['D1'])
    # left_undistort = cv2.undistort(left_o, calib['M1'], calib['D1'])
    # right_undistort = cv2.undistort(right_o, calib['M2'], calib['D2'])

    cv2.imwrite('../data/mask_und.png', mask_und)
    # cv2.imwrite('../data/left_und.bmp', left_undistort)
    # cv2.imwrite('../data/right_und.bmp', right_undistort)
    exit()

    # try to multiply h with p

    # x3 = np.asarray([14.9, 1.338, 59., 1.]).reshape(-1, 1)
    x3 = np.asarray([9.3125, 6.8502, 68.6674, 1.]).reshape(-1, 1)# 388 354 , [359, 353]
    # H1 = np.asarray([0.9393039942, 0.0054876450, -30.5864753723, -0.0418271311,
    #                  0.9963799119, -4.1946434975, -0.0001156561, -0.0000000000, 1.0000000000]).reshape(3, 3)

# [[394, 327], [350, 347], [313, 407], [369, 364]]
# [[430, 332], [386, 352], [351, 409], [404, 367]]
# [[398, 328], [354, 348], [317, 408], [373, 365]]

# [[-9.85915572e-01 -7.36637438e-01  4.96982131e+02]
#  [-6.11799087e-01 -9.65028229e-01  4.54427296e+02]
#  [-1.74922946e-03 -1.89171888e-03  1.00000000e+00]]

    # 430 332 9.893020 2.413177 66.415535 - triangulate([[ 9.76994039], [ 2.63494899],[66.41993952]])
    # -> 404
    # m1 = np.asarray([394., 327.]).reshape(2,1)
    # m2 = np.asarray([398., 328.]).reshape(2,1)
    x3 = np.asarray([9.76994039, 2.63494899 , 66.41993952, 1.]).reshape(-1, 1)

    # 386 352 3.067096 5.422349 66.734306 - triangulate([[ 3.03488394], [ 5.32636311],[64.71968012]])
    # m1 = np.asarray([350., 347.]).reshape(2,1)
    # m2 = np.asarray([354., 348.]).reshape(2,1)
    # x3 = np.asarray([3.067096, 5.422349, 66.734306, 1.]).reshape(-1, 1)

    # 351 409 -2.330986 13.903185 66.937828 - triangulate([[-2.37987674], [13.39995303], [63.68467712]]
    # m1 = np.asarray([313., 407.]).reshape(2,1)
    # m2 = np.asarray([317., 408.]).reshape(2,1)
    # x3 = np.asarray([-2.37987674,13.39995303,63.68467712, 1.]).reshape(-1, 1)

    # 404 367 6.017776 7.818072 68.197891 - triangulate([[ 5.8726539 ], [ 7.74571407], [65.13614288]]
    # m1 = np.asarray([369., 364.]).reshape(2,1)
    # m2 = np.asarray([373., 365.]).reshape(2,1)
    # x3 = np.asarray([5.8726539 , 7.74571407,65.13614288, 1.]).reshape(-1, 1)

    # points_4d = cv2.triangulatePoints(P1, P2, m1, m2)
    # points_3d = (points_4d/points_4d[3])[:3]

    # print(points_3d)


    # exit()



    H1 = np.asarray([0.9393039942, 0.0054876450, -30.5864753723, -0.0418271311,
                     0.9963799119, -4.1946434975, -0.0001156561, -0.0000000000, 1.0000000000]).reshape(3, 3)

    H2 = np.asarray([0.9643145800, -0.0017762216,   0.0000018923,  -0.0256799851,
                     0.9932777882, -2.0000000000, -0.0000488560,  -0.0000298142,  1.0000000000]).reshape(3, 3)



    

    # left_wraped_rect = cv2.warpPerspective(
    #     left_und_img, np.linalg.inv(H1), dsize=left.shape[:2][::-1])
    # cv2.imwrite('./wrapped_to_rect.png', left_wraped_rect)
    # left_wraped_und = cv2.warpPerspective(left, H1, dsize=left.shape[:2][::-1])
    # cv2.imwrite('./wrapped_to_undistort.png', left_wraped_und)


    # left_wraped_o_rect = cv2.warpPerspective(left_o, np.linalg.inv(H1), dsize=left.shape[:2][::-1])
    # cv2.imwrite('./wrapped_to_undistortfrom0.png', left_wraped_o_rect)
    # left_wraped_rect_o = cv2.warpPerspective(left, H1, dsize=left.shape[:2][::-1])
    # cv2.imwrite('./wrapped_to_original.png', left_wraped_o_rect)
    # exit()
    n1, Rs1, Ts1, Ns1 = cv2.decomposeHomographyMat(np.linalg.inv(H1), calib['M1'])
    n2, Rs2, Ts2, Ns2 = cv2.decomposeHomographyMat(np.linalg.inv(H2), calib['M2'])
    Rt__l = np.eye(4)
    Rt__r = np.eye(4)
    Rt__r[:3, :3] = calib['R'].copy()
    Rt__r[:3, 3] = calib['T'].copy()
    K1 = calib['M1']
    K2 = calib['M2']
    for i in range(n1):
        for j in range(n2):
            Rl = Rs1[i].copy()
            Rr = Rs2[j].copy()
            Tl = Ts1[i].copy()
            Tr = Ts2[j].copy()
            RT1s = np.eye(4)
            RT2s = np.eye(4)
            RT1s[:3, :3] = Rl
            RT1s[:3, 3] = Tl.reshape(-1)
            RT2s[:3,:3]= Rr
            RT2s[:3,3] = Tr.reshape(-1)

            # P1_test= calib['M1'].dot(RT1s.dot(Rt__l)[:3])
            # # P2_test= calib['M2'].dot(RT2s.dot(Rt__r)[:3])

            # P1_test= calib['M1'].dot(np.linalg.inv(Rt__l).dot(RT1s)[:3])
            # # P2_test= calib['M2'].dot(np.linalg.inv(Rt__r).dot(RT2s)[:3])

            # P1_test= calib['M1'].dot(Rt__l.dot(RT1s)[:3])
            # # P2_test= calib['M2'].dot(Rt__r.dot(RT2s)[:3])

            # P1_test= calib['M1'].dot(np.linalg.inv(RT1s).dot(Rt__l)[:3])
            # # P2_test= calib['M2'].dot(np.linalg.inv(RT2s).dot(Rt__r)[:3])

            # P1_test= calib['M1'].dot((RT1s)[:3])
            # # P2_test= calib['M2'].dot((RT2s)[:3])



            P1_test= calib['M1'].dot(RT1s[:3])
            P2_test= calib['M2'].dot(RT2s[:3])


            P1_test = K1.dot(RT1s[:3])
            pix1 = P1_test.dot(x3)

            # print(Rt__r)
            # print(np.linalg.inv(Rt__r))

            RTs2_n =  RT2s * (Rt__r)
            RTs2_n =  RT2s * np.linalg.inv(Rt__r)
            RTs2_n =  np.linalg.inv(RT2s) * (Rt__r)
            RTs2_n =  Rt__r * RT2s
            RTs2_n =  np.linalg.inv(Rt__r) * RT2s 
            RTs2_n =  Rt__r * np.linalg.inv(RT2s)
            RTs2_n =  RT2s
            RTs2_n =  Rt__r
            RTs2_n =  np.linalg.inv(Rt__r)
            RTs2_n =  np.linalg.inv(RT2s)


            scene_point = x3[:3].reshape(3, 1)
            img_points_l = cv2.projectPoints(scene_point, cv2.Rodrigues(Rl)[0], np.asarray(Tl), calib['M1'], None)
            img_points_r = cv2.projectPoints(scene_point, cv2.Rodrigues(RTs2_n[:3, :3])[0], np.asarray(RTs2_n[:3, 3] ), calib['M2'], None)
            diff_x  = (img_points_l[0] - img_points_r[0])[0,0,0]
            diff_y = (img_points_l[0] - img_points_r[0])[0,0,1]
            print(diff_x, diff_y)
            if (diff_x > 0):
                if abs(diff_y) <1.5:
                    print(i,j,img_points_l[0], img_points_r[0])
            # pix2 = P2_test.dot(x3)
            # print((pix1[0]/pix1[2]).item())
            # print(pix1/pix1[2]-pix2/pix2[2])
    exit()

    # undist_l = cv2.undistort(left, calib['M1'], calib['D1'])
    # undist_r = cv2.undistort(right, calib['M2'], calib['D2'])

    # distort_m1 = np.asarray([221., 318.]).reshape(2,1)
    # distort_m2 = np.asarray([226., 321.]).reshape(2,1)

    # undist_lp = np.ones((3,1))
    # undist_rp = np.ones((3,1))
    # undist_lp[:2] = cv2.undistortPoints(distort_m1, calib['M1'], calib['D1']).reshape(2,1)
    # undist_rp[:2] = cv2.undistortPoints(distort_m2, calib['M2'], calib['D2']).reshape(2,1)

    # print(calib['M1'].dot(undist_lp))
    # print(calib['M2'].dot(undist_rp))

    # cv2.imwrite('./undistort_left.png', undist_l)
    # cv2.imwrite('./undistort_right.png', undist_r)

    # m1 = np.asarray([255., 326.]).reshape(2,1)
    # m2 = np.asarray([218., 326.]).reshape(2,1)
    # m1 = np.asarray([218.42, 318.26]).reshape(2,1)
    # m2 = np.asarray([220., 321.9]).reshape(2,1)
    # points_4d = cv2.triangulatePoints(P1, P2, m1, m2)
    # points_3d = (points_4d/points_4d[3])[:3]

    # print(points_3d)

    # exit()

    # P1n = np.linalg.inv(H1).dot(P1)
    # P2n = np.linalg.inv(H2).dot(P2)
    # # P1n = (H1).dot(P1)
    # # P2n = (H2).dot(P2)

    # pix1 = P1.dot(x3)
    # pix2 = P2.dot(x3)
    # print(pix1[:2]/pix1[2])
    # print(pix1/pix1[2]-pix2/pix2[2])

    # exit()
    # print(H1)

    rt1 = np.eye(4)
    rt2 = rt1.copy()
    rt1[:2, 3] = 1
    rt2[:2, 3] = 2
    print(rt1.dot(rt2))

    point_left = np.asarray([[432], [433]]).reshape(2, 1).astype('float')
    point_right = np.asarray([[437], [433]]).reshape(2, 1).astype('float')

    _, Rs1, Ts1, Ns1 = cv2.decomposeHomographyMat(H1, calib['M1'])
    _, Rs2, Ts2, Ns2 = cv2.decomposeHomographyMat(H2, calib['M2'])
    # 371 350 0.719575 5.048121 65.672897
    # x3 = np.asarray([0.719575, 5.048121, 65.672897, 1]).reshape(-1,1)
    # 399 366 5.209150 7.653638 68.087944
    # x3 = np.asarray([5.209150, 7.653638, 68.087944, 1]).reshape(-1,1)
    # x3 = np.asarray([-6.86141, 11.8293, 63.9924, 1]).reshape(-1,1)

    # numpy indexing y,x
    known_disps = np.nonzero(d)
    print(known_disps)
    pl = list(zip(*known_disps))
    pr = list((y, x + d[y, x]) for y, x in pl)
    print(pl[0])
    pr = list((x + d[y, x], y) for x, y in pl)
    pl = np.asarray([list(a) for a in pl]).transpose()
    pr = np.asarray([list(a) for a in pr]).transpose()

    Rt__l = np.eye(4)
    Rt__r = np.eye(4)
    Rt__r[:3, :3] = calib['R'].copy()
    Rt__r[:3, 3] = calib['T'].copy()
    for i in range(4):
        for j in range(4):
            Rl = Rs1[i].copy()
            Rr = Rs2[j].copy()
            Tl = Ts1[i].copy()
            Tr = Ts2[j].copy()
            RT1s = np.eye(4)
            RT2s = np.eye(4)
            RT1s[:3, :3] = Rl
            RT1s[:3, 3] = Tl.reshape(-1)
            RT2s[:3, :3] = Rr
            RT2s[:3, 3] = Tr.reshape(-1)
            # print(i,j)

            # P1_test= calib['M1'].dot(Rt__l.dot(RT1s)[:3])
            # P2_test= calib['M2'].dot(Rt__r.dot(RT2s)[:3])
            P1_test = calib['M1'].dot(np.linalg.inv(RT1s).dot(Rt__l)[:3])
            P2_test = calib['M2'].dot(np.linalg.inv(RT2s).dot(Rt__r)[:3])
            pix1 = P1_test.dot(x3)
            pix2 = P2_test.dot(x3)
            print(pix1[:2]/pix1[2])
            print(pix1/pix1[2]-pix2/pix2[2])
            # P1= P1_test
            # P2 = P2_test
            # scene_points = []
            # for k in range(pl.shape[1]):
            #     point_left = pl[:, k].reshape(2, 1)
            #     point_right = pr[:, k].reshape(2, 1)
            #     # print(point_left, point_right)

            #     points_4d = cv2.triangulatePoints(
            #         P1, P2, point_left.astype('float'), point_right.astype('float'))
            #     points_3d = (points_4d/points_4d[3])[:3]

            #     entry = [point_left[0], point_left[1], *points_3d]

            #     entry = [a.item() for a in entry]

            #     scene_points.append(entry)

            # # create_xyz(scene_points, './sdf.txt')
            # create_pcd(scene_points, left, './reconstruction_{}_{}.ply'.format(i,j))

            # points_4d = cv2.triangulatePoints(P1_test, P2_test, point_left, point_right)
            # points_3d = (points_4d/points_4d[3])[:3]

            # entry = [point_left[1], point_left[0], *points_3d]
            # print(points_3d)

            # a point should lie in the same scanline

    exit()
    # print(Rs)
    # print(Ts)
    # print(Ns)

    # pix1 = P1.dot(x3)
    # pix2 = P2.dot(x3)
    # print('pixel_left\n', pix1/pix1[2])
    # print('pixel_right\n', pix2/pix2[2])
    # points_4d = cv2.triangulatePoints(P1, P2, point_left, point_right)
    # points_3d = (points_4d/points_4d[3])[:3]

    # entry = [point_left[1], point_left[0], *points_3d]
    # print(points_3d)

    # numpy indexing y,x
    known_disps = np.nonzero(d)
    # print(known_disps)
    # pl = list(zip(known_disps[1], known_disps[0]))
    pl = list(zip(*known_disps))
    pr = list((y, x + d[y, x]) for y, x in pl)
    # print(pl[0])
    pr = list((x + d[y, x], y) for x, y in pl)
    pl = np.asarray([list(a) for a in pl]).transpose()
    pr = np.asarray([list(a) for a in pr]).transpose()

    scene_points = []
    for i in range(pl.shape[1]):
        point_left = pl[:, i].reshape(2, 1)
        point_right = pr[:, i].reshape(2, 1)
        print(point_left, point_right)

        points_4d = cv2.triangulatePoints(
            P1, P2, point_left.astype('float'), point_right.astype('float'))
        points_3d = (points_4d/points_4d[3])[:3]

        entry = [point_left[0], point_left[1], *points_3d]

        entry = [a.item() for a in entry]

        scene_points.append(entry)
        break

    create_xyz(scene_points, './sdf.txt')
    create_pcd(scene_points, left, './reconstruction.ply')

    # cv2.imwrite('./rect.png',rect)
    # cv2.imwrite('./rect_eval.png',rect_eval)
