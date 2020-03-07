import random
import cv2
import numpy as np
import dlib
import torchvision
import torchvision.transforms as transforms

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

IM_SIZE = 224



def get_landmark(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    pos = []
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # pos = (point[0, 0], point[0, 1])
            a = (point[0, 0], point[0, 1])
            a = list(a)
            pos += a
    pos = np.array(pos, dtype=int)
    landmarks = np.reshape(pos, (68, -1))
    return landmarks


def get_au_tg_dlib(array_68, h, w):
    str_dt = list(array_68[:, 0]) + list(array_68[:, 1])
    region_array = np.zeros((11, 4))
    # print str_dt
    try:
        # print len(str_dt)
        W = w
        H = h
        arr2d = np.array(str_dt).reshape((2, 68))
        # print arr2d
        arr2d[0, :] = arr2d[0, :] / W * 100
        arr2d[1, :] = arr2d[1, :] / H * 100
        region_bbox = []
        ruler = abs(arr2d[0, 39] - arr2d[0, 42])
        # print ruler
        region_bbox += [[arr2d[0, 21], arr2d[1, 21] - ruler / 2, arr2d[0, 22], arr2d[1, 22] - ruler / 2]]  # 0
        # region_bbox+=[[arr2d[0,21]/2+arr2d[0,22]/2,arr2d[1,21]/2+arr2d[1,22]/2,arr2d[0,39]/2+arr2d[0,42]/2,arr2d[1,39]/2+arr2d[1,42]/2]]
        region_bbox += [[arr2d[0, 18], arr2d[1, 18] - ruler / 3, arr2d[0, 25], arr2d[1, 25] - ruler / 3]]  # 2
        # replace layers

        region_bbox += [[arr2d[0, 19], arr2d[1, 19] + ruler / 3, arr2d[0, 24], arr2d[1, 24] + ruler / 3]]  # 3: au4
        region_bbox += [[arr2d[0, 41], arr2d[1, 41] + ruler, arr2d[0, 46], arr2d[1, 46] + ruler]]  # 4: au6
        region_bbox += [[arr2d[0, 38], arr2d[1, 38], arr2d[0, 43], arr2d[1, 43]]]  # 5: au7
        region_bbox += [[arr2d[0, 49], arr2d[1, 49], arr2d[0, 53], arr2d[1, 53]]]  # 6: au10
        region_bbox += [[arr2d[0, 48], arr2d[1, 48], arr2d[0, 54], arr2d[1, 54]]]  # 7: au12 au14 lip corner
        region_bbox += [[arr2d[0, 51], arr2d[1, 51], arr2d[0, 57], arr2d[1, 57]]]  # 8: au17
        region_bbox += [[arr2d[0, 61], arr2d[1, 61], arr2d[0, 63], arr2d[1, 63]]]  # 9: au 23 24
        region_bbox += [[arr2d[0, 56], arr2d[1, 56] + ruler / 2, arr2d[0, 58], arr2d[1, 58] + ruler / 2]]  # 10: #au23

        region_array = np.array(region_bbox)
    except Exception as e:
        i = 0;
    return region_array


def get_map(array_68, h, w):
    feat_map = np.zeros((100, 100))
    tg_array = get_au_tg_dlib(array_68, h, w)
    for i in range(tg_array.shape[0]):
        for j in range(2):
            pt = tg_array[i, j * 2:(j + 1) * 2]
            pt = pt.astype('uint8')
            # print pt
            for px in range(pt[0] - 7, pt[0] + 7):
                if px < 0 or px > 99:
                    break
                for py in range(pt[1] - 7, pt[1] + 7):
                    if py < 0 or py > 99:
                        break
                    d1 = abs(px - pt[0])
                    d2 = abs(py - pt[1])
                    value = 1 - (d1 + d2) * 0.07
                    feat_map[py][px] = max(feat_map[py][px], value)
                # print feat_map[py][px]
    # feat_map=cv2.resize(feat_map,(224,224))
    # print value
    return feat_map


def feat_data(imglist):
    # datablob = np.ndarray((data_size, 4, IM_SIZE, IM_SIZE))
    # fls = imglist[:data_size]
    fls = imglist
    feat_map224 = np.zeros((IM_SIZE, IM_SIZE))

    # im224 = np.zeros((4, IM_SIZE, IM_SIZE))
    imi = cv2.imread(fls)
    landmark = get_landmark(imi)
    if np.shape(landmark)[1] != 0:

        # print(np.shape(landmark)[1])

        feat_map = get_map(landmark, imi.shape[0], imi.shape[1])

        # for t in range(3):
        #     im224[t, :, :] = cv2.resize(imi[:, :, t], (IM_SIZE, IM_SIZE))
        feat_map224 = cv2.resize(feat_map, (224, 224))
        # im224[3, :, :] = feat_map224
    return feat_map224

        # datablob[i, :, :, :] = im224  # (2, 4, 224, 224)
    # return datablob





# feat_data = feat_data('./RAF/bbbb/train_00001_aligned.jpg')
# print(np.shape(feat_data))




#
# def feat_data(imglist, data_size):
#     datablob = np.ndarray((data_size, 4, IM_SIZE, IM_SIZE))
#     fls = imglist[:data_size]
#     # fls = imglist
#
#     im224 = np.zeros((4, IM_SIZE, IM_SIZE))
#     for i in range(len(fls)):
#         path = './bbbb/'
#         a = path + fls[i].split('.')[0] + '_aligned.jpg'
#         # p.append(a)
#         imi = cv2.imread(a)
#         landmark = get_landmark(imi)
#         if np.shape(landmark)[1] == 0:
#             continue
#
#         # print(np.shape(landmark)[1])
#         for t in range(3):
#             im224[t, :, :] = cv2.resize(imi[:, :, t], (IM_SIZE, IM_SIZE))
#         # print(np.shape(im224))  # (4, 224, 224)
#         feat_map = get_map(landmark, imi.shape[0], imi.shape[1])
#         feat_map224 = cv2.resize(feat_map, (224, 224))
#         im224[3, :, :] = feat_map224
#         datablob[i, :, :, :] = im224  # (2, 4, 224, 224)
#         # datablob[0, :, :, :] = im224  # (2, 4, 224, 224)
#     return datablob
#     # return im224
#
# listtrainpath = 'bbbb.txt'
# fp = open(listtrainpath)
# imglist = fp.readlines()
# # p = []
# feat_data = feat_data(imglist, BATCH_SIZE)
# print(np.shape(feat_data), feat_data)






