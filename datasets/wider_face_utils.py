import math
from operator import ne, pos
import os
import cv2
import numpy as np
from random import random
from PIL import Image
from progress.bar import Bar


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def generate_annotations(ann_path, data_path, save_path):
    cate_list = os.listdir(data_path)
    bar = Bar('generating ground truth:', max=12879)

    count = 1
    with open(ann_path) as f:
        line = f.readline().replace('\n', '')
        cate = line.split('/')[0]
        save_path_cate = make_sure_path_exists(os.path.join(save_path, cate))
        img_name = line.split('/')[1].replace('.jpg', '')
        bbox_list = []
        while line:
            line = f.readline().replace('\n', '')
            if line == '':
                gt_bboxs = np.vstack(bbox_list[1:])
                np.save(os.path.join(save_path_cate, img_name), gt_bboxs)
                break
            cate = line.split('/')[0]

            if cate in cate_list:
                gt_bboxs = np.vstack(bbox_list[1:])
                np.save(os.path.join(save_path_cate, img_name), gt_bboxs)
                bar.suffix = f'{count} / 12879'
                bar.next()
                count += 1

                save_path_cate = make_sure_path_exists(os.path.join(save_path, cate))
                img_name = line.split('/')[1].replace('.jpg', '')
                bbox_list = []
            else:
                gts = np.array(line.strip().split(' ')).astype('int')
                bbox_list.append(gts)
        
        bar.finish()


def cal_iou(box1, box2):
    inter_x1, inter_y1 = np.maximum(box1[:, 0], box2[:, 0]), np.maximum(box1[:, 1], box2[:, 1])
    inter_x2, inter_y2 = np.minimum(box1[:, 0] + box1[:, 2], box2[:, 0] + box2[:, 2]), np.minimum(box1[:, 1] + box1[:, 3], box2[:, 1] + box2[:, 3])

    inter = np.maximum((inter_x2 - inter_x1), 0) * np.maximum((inter_y2 - inter_y1), 0)
    return inter / (box1[:, 2] * box1[:, 3] + box2[:, 2] * box2[:, 3] - inter)


# WIDER FACE ground truth: (col, row, width, height)
def generate_offset(bbox):
    offset = np.random.uniform(0, 1, 4)
    offset[:2] = offset[:2] - 0.5
    offset[2:] = offset[2:] * 2 + 1e-7

    if bbox[2] > bbox[3]:
        side = math.ceil(bbox[2] / offset[2])
        offset[2] = math.log(offset[2])
        offset[3] = math.log(bbox[3] / side)
    else:
        side = math.ceil(bbox[3] / offset[3])
        offset[3] = math.log(offset[3])
        offset[2] = math.log(bbox[2] / side)

    delta_x = int(offset[0] * side)
    delta_y = int(offset[1] * side)

    x = bbox[0] - delta_x
    y = bbox[1] - delta_y

    return offset, x, y, side

# shape of Image readed from Pillow is (rows, columns, channels)
def generate_set(img_path, gt_path, save_path, folder='img', label_name='label_list_data', mode='train', part=False, ratio=[1, 2, 2]):
    cate_list = os.listdir(img_path)
    bar = Bar(f'generating {mode}set:', max=999999)

    make_sure_path_exists(os.path.join(save_path, 'img'))

    faces_count = 30360
    label_list = []

    for cate in cate_list:
        img_list = os.listdir(os.path.join(img_path, cate))
        for img_name in img_list:
            img = np.array(Image.open(os.path.join(img_path, cate, img_name)))
            gts = np.load(os.path.join(gt_path, cate, img_name.replace('jpg', 'npy')))
            img_name = img_name.replace('.jpg', '')

            img_h, img_w = img.shape[:2]
            min_side = 24 if part else 12

            for bbox in gts:
                if bbox[2] < min_side or bbox[3] < min_side or min(bbox[2],bbox[3]) / max(bbox[2],bbox[3]) <= 2/3 or bbox[-3] == 1:
                    continue

                pos_count, par_count, neg_count = 0, 0, 0

                # positive and part samples
                while True:
                    offset, x, y, side = generate_offset(bbox)

                    if x < 0 or y < 0 or side < 12 or x + side >= img_w or y + side >= img_h:
                        continue

                    iou = cal_iou(np.array([[x, y, side, side]]), bbox[:4].reshape(1, -1))
                    face = img[y:y + side, x:x + side]

                    label = [0, 0, 0, 0, 0]

                    if iou >= 0.6 and pos_count < ratio[0]:
                        label = [offset[0], offset[1], offset[2], offset[3], 1]
                        pos_count += 1

                    elif iou > 0.3 and iou < 0.6 and par_count < ratio[1] and part == True:
                        label = [offset[0], offset[1], offset[2], offset[3], -1]
                        par_count += 1

                    else:
                        continue

                    # print(label, bbox[:4], (x, y, side, side), img_name, faces_count)
                    np.save(os.path.join(save_path, folder, f'{faces_count}'), face)
                    label = label + bbox[:4].tolist()
                    label.append(x)
                    label.append(y)
                    label_list.append(np.array(label))

                    bar.suffix = f'{faces_count} / unknown'
                    bar.next()

                    faces_count += 1
                    if pos_count >= ratio[0] and par_count * part >= ratio[1] * part:
                        break

                # negitive samples
                while neg_count < ratio[2]:
                    side = int(np.random.uniform(2) * min(bbox[2:4]))
                    if side >= min(img_w, img_h) * 0.7:
                        break

                    x = np.random.randint(0, img_w - side)
                    y = np.random.randint(0, img_h - side)

                    if x < 0 or y < 0 or side < 12 or x + side >= img_w or y + side >= img_h:
                        continue

                    face = img[y:y + side, x:x + side]


                    label = [0, 0, 0, 0, 0]
                    iou = cal_iou(np.array([[x, y, side, side]]), bbox[:4].reshape(1, -1))

                    if (iou > 0.3).sum() == 0 and neg_count < ratio[2]:
                        neg_count += 1
                    else:
                        continue

                    np.save(os.path.join(save_path, folder, f'{faces_count}'), face)
                    label = label + bbox[:4].tolist()
                    label.append(x)
                    label.append(y)
                    label_list.append(np.array(label))

                    bar.suffix = f'{faces_count} / unknown'
                    bar.next()

                    faces_count += 1
        #     if faces_count > 20:
        #         break
        # break

    bar.finish()
    np.save(os.path.join(save_path, label_name), np.vstack(label_list))


def resize_img(img_path, save_path, size=(12, 12)):
    file_list = os.listdir(img_path)
    num_imgs = len(file_list)

    bar = Bar('resizing images:', max=num_imgs)

    make_sure_path_exists(save_path)

    for idx, file in enumerate(file_list):
        img = Image.fromarray(np.load(os.path.join(img_path, file)))
        img = np.array(img.resize(size))

        np.save(os.path.join(save_path, file), img)
        bar.suffix = f'{idx + 1} / {num_imgs}'
        bar.next()
    bar.finish()


def mean_std(path):
    img_list = os.listdir(path)
    pixels_num = 0
    value_sum = [0, 0, 0]
    files_num = len(img_list)
    bar = Bar('Calculating mean:', max=files_num)

    for idx, img_file in enumerate(img_list):
        img = np.load(os.path.join(path, img_file)) / 255.0
        pixels_num += img.shape[0] * img.shape[1]
        value_sum += np.sum(img, axis=(0, 1))
        bar.suffix = f'{idx + 1} / {files_num}'
        bar.next()
    bar.finish()

    value_mean = value_sum / pixels_num
    value_std = _std(path, img_list, value_mean, pixels_num)
    return value_mean, value_std


def _std(path, img_list, mean, pixels_num):
    value_std = [0, 0, 0]
    files_num = len(img_list)
    bar = Bar('Calculating std:', max=files_num)

    for idx, img_file in enumerate(img_list):
        img = np.load(os.path.join(path, img_file)) / 255.0
        value_std += np.sum((img - mean) ** 2, axis=(0, 1))
        bar.suffix = f'{idx + 1} / {files_num}'
        bar.next()
    bar.finish()
    return np.sqrt(value_std / pixels_num)


def img2nparray(img_path, save_path):
    make_sure_path_exists(save_path)
    img_list = os.listdir(img_path)
    num_imgs = len(img_list)
    bar = Bar('transpose image to numpy array:', max=num_imgs)

    for idx, img_file in enumerate(img_list):
        img = Image.open(os.path.join(img_path, img_file))
        img = np.array(img)

        np.save(os.path.join(save_path, img_file.split('.')[0]), img)
        bar.suffix = f'{idx} / {num_imgs}'
        bar.next()
    bar.finish()


if __name__ == '__main__':
    dataset = 'val'
    model = 'onet'

    if model == 'pnet':
        img_size = '12x12'
        size = (12, 12)
    elif model == 'rnet':
        img_size = '24x24'
        size = (24, 24)
    elif model == 'onet':
        img_size = '48x48'
        size = (48, 48)


    # ann_path = f'/data/grey/WIDER_FACE/wider_face_split/wider_face_{dataset}_bbx_gt.txt'
    # data_path = f'/data/grey/WIDER_FACE/WIDER_{dataset}/images/'
    # save_path = f'/data/grey/WIDER_FACE/WIDER_{dataset}/labels/'

    # generate_annotations(ann_path, data_path, save_path)


    img_path = f'/data/grey/WIDER_FACE/WIDER_{dataset}/images/'
    gt_path = f'/data/grey/WIDER_FACE/WIDER_{dataset}/labels/'
    save_path = f'/data/grey/WIDER_FACE/{model}_{dataset}set/'

    generate_set(img_path, gt_path, save_path, 'img', 'label_list_data', mode=dataset, part=True, ratio=[3, 3, 6])


    img_path = f'/data/grey/WIDER_FACE/{model}_{dataset}set/img'
    save_path = f'/data/grey/WIDER_FACE/{model}_{dataset}set/img{img_size}'

    resize_img(img_path, save_path, size)

    # img_path = f'/data/grey/WIDER_FACE/{model}_{dataset}set/img{img_size}'
    # save_path = f'/data/grey/WIDER_FACE/{model}_{dataset}set/np{img_size}'
    # img2nparray(img_path, save_path)
