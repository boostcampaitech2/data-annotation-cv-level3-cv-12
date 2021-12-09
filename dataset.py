import os.path as osp
import math
import json
from PIL import Image

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon

import numpy.random as npr
from albumentations.pytorch import ToTensorV2
from shapely.geometry import Polygon


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels


class SceneTextDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=1024, transform = None):
        # root로 받아온 anno를 for문을 돌면서 모들 json을 받아와 줌
        # 마찬가지로 사용하시는 분의 디렉토리 구조와 동일하게 만드는 작업을 해주셔야 합니다.
        annos = []
        image_dir = []
        for dir in root_dir :
            image_dir.append(osp.join(dir, 'images'))
            with open(osp.join(dir, 'ufo/{}.json'.format(split)), 'r') as f:
                annos.append(json.load(f))

        self.annos = annos
        self.image_fnames = []
        # 받아온 이미지 리스트를 여기서 1개로 합쳐줍니다.
        for anno in annos :
            self.image_fnames.extend(sorted(anno['images'].keys()))
        
        self.image_dir = image_dir
        # self.image_dir = osp.join(root_dir, 'images')

        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        # 합쳐진 이미지 중 1개를 뽑고
        image_fname = self.image_fnames[idx]

        # 그 이미지가 속해 있는 폴더를 찾아가게 합니다.
        for i, anno in enumerate(self.annos) :
            if image_fname in anno['images'].keys() :
                image_fpath = osp.join(self.image_dir[i], image_fname)
                vertices, labels = [], []
                for word_info in anno['images'][image_fname]['words'].values():
                    vertices.append(np.array(word_info['points']).flatten())
                    labels.append(int(not word_info['illegibility']))
                vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)
                vertices, labels = filter_vertices(vertices, labels, ignore_under=10, drop_under=1)
                break
            else :
                continue
        
        
        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        if self.transform is not None:
            transformed = self.transform(image=image,word_bboxes=word_bboxes)

        image = transformed['image']
        word_bboxes = transformed['word_bboxes']
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask




def transform_by_matrix(matrix, image=None, oh=None, ow=None, word_bboxes=[],
                        by_word_char_bboxes=[], masks=[], inverse=False):
    """
    Args:
        matrix (ndarray): (3, 3) shaped transformation matrix.
        image (ndarray): (H, W, C) shaped ndarray.
        oh (int): Output height.
        ow (int): Output width.
        word_bboxes (List[ndarray]): List of (N, 2) shaped ndarrays.
        by_word_char_bboxes (List[ndarray]): Lists of (N, 4, 2) shaped ndarrays.
        masks (List[ndarray]): List of (H, W) shaped ndarray the same size as the image.
        inverse (bool): Whether to apply inverse transformation.
    """
    if image is not None or masks is not None:
        assert oh is not None and ow is not None

    output_dict = dict()

    if inverse:
        matrix = np.linalg.pinv(matrix)

    if image is not None:
        output_dict['image'] = cv2.warpPerspective(image, matrix, dsize=(ow, oh))

    if word_bboxes is None:
        output_dict['word_bboxes'] = None
    elif len(word_bboxes) > 0:
        num_points = list(map(len, word_bboxes))
        points = np.concatenate([np.reshape(bbox, (-1, 2)) for bbox in word_bboxes])  # (N, 2)
        points = cv2.perspectiveTransform(
            np.reshape(points, (1, -1, 2)).astype(np.float32), matrix).reshape(-1, 2)  # (N, 2)
        output_dict['word_bboxes'] = [
            points[i:i + n] for i, n in zip(np.cumsum([0] + num_points)[:-1], num_points)]
    else:
        output_dict['word_bboxes'] = []

    if by_word_char_bboxes is None:
        output_dict['by_word_char_bboxes'] = None
    elif len(by_word_char_bboxes) > 0:
        word_lens = list(map(len, by_word_char_bboxes))
        points = np.concatenate([np.reshape(bboxes, (-1, 2)) for bboxes in by_word_char_bboxes])  # (N, 2)
        points = cv2.perspectiveTransform(
            np.reshape(points, (1, -1, 2)).astype(np.float32), matrix).reshape(-1, 4, 2)  # (N, 4, 2)
        output_dict['by_word_char_bboxes'] = [
            points[i:i + n] for i, n in zip(np.cumsum([0] + word_lens)[:-1], word_lens)]
    else:
        output_dict['by_word_char_bboxes'] = []

    if masks is None:
        output_dict['masks'] = None
    else:
        output_dict['masks'] = [cv2.warpPerspective(mask, matrix, dsize=(ow, oh)) for mask in masks]

    return output_dict


class GeoTransformation:
    """
    Args:
    """
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        random_translate=False, min_image_overlap=0, min_bbox_overlap=0, min_bbox_count=0,
        allow_partial_occurrence=True,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100
    ):
        if rotate_anchors is None:
            self.rotate_anchors = None
        elif type(rotate_anchors) in [int, float]:
            self.rotate_anchors = [rotate_anchors]
        else:
            self.rotate_anchors = list(rotate_anchors)

        if rotate_range is None:
            self.rotate_range = None
        elif type(rotate_range) in [int, float]:
            assert rotate_range >= 0
            self.rotate_range = (-rotate_range, rotate_range)
        elif len(rotate_range) == 2:
            assert rotate_range[0] <= rotate_range[1]
            self.rotate_range = tuple(rotate_range)
        else:
            raise TypeError

        if crop_aspect_ratio is None:
            self.crop_aspect_ratio = None
        elif type(crop_aspect_ratio) in [int, float]:
            self.crop_aspect_ratio = float(crop_aspect_ratio)
        elif len(crop_aspect_ratio) == 2:
            self.crop_aspect_ratio = tuple(crop_aspect_ratio)
        else:
            raise TypeError

        if type(crop_size) in [int, float]:
            self.crop_size = crop_size
        elif len(crop_size) == 2:
            assert type(crop_size[0]) == type(crop_size[1])
            self.crop_size = tuple(crop_size)
        else:
            raise TypeError

        assert crop_size_by in ['width', 'height', 'longest']
        self.crop_size_by = crop_size_by

        self.hflip, self.vflip = hflip, vflip

        self.random_translate = random_translate

        self.min_image_overlap = max(min_image_overlap or 0, 0)
        self.min_bbox_overlap = max(min_bbox_overlap or 0, 0)
        self.min_bbox_count = max(min_bbox_count or 0, 0)
        self.allow_partial_occurrence = allow_partial_occurrence

        self.max_random_trials = max_random_trials

        if resize_to is None:
            self.resize_to = resize_to
        elif type(resize_to) in [int, float]:
            if not keep_aspect_ratio:
                self.resize_to = (resize_to, resize_to)
            else:
                self.resize_to = resize_to
        elif len(resize_to) == 2:
            assert not keep_aspect_ratio
            assert type(resize_to[0]) == type(resize_to[1])
            self.resize_to = tuple(resize_to)
        assert resize_based_on in ['width', 'height', 'longest']
        self.keep_aspect_ratio, self.resize_based_on = keep_aspect_ratio, resize_based_on

    def __call__(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[]):
        return self.crop_rotate_resize(image, word_bboxes=word_bboxes,
                                       by_word_char_bboxes=by_word_char_bboxes, masks=masks)

    def _get_theta(self):
        if self.rotate_anchors is None:
            theta = 0
        else:
            theta = npr.choice(self.rotate_anchors)
        if self.rotate_range is not None:
            theta += npr.uniform(*self.rotate_range)

        return theta

    def _get_patch_size(self, ih, iw):
        if (self.crop_aspect_ratio is None and isinstance(self.crop_size, float) and
            self.crop_size == 1.0):
            return ih, iw

        if self.crop_aspect_ratio is None:
            aspect_ratio = iw / ih
        elif isinstance(self.crop_aspect_ratio, float):
            aspect_ratio = self.crop_aspect_ratio
        else:
            aspect_ratio = np.exp(npr.uniform(*np.log(self.crop_aspect_ratio)))

        if isinstance(self.crop_size, tuple):
            if isinstance(self.crop_size[0], int):
                crop_size = npr.randint(self.crop_size[0], self.crop_size[1])
            elif self.crop_size[0]:
                crop_size = np.exp(npr.uniform(*np.log(self.crop_size)))
        else:
            crop_size = self.crop_size

        if self.crop_size_by == 'longest' and iw >= ih or self.crop_size_by == 'width':
            if isinstance(crop_size, int):
                pw = crop_size
                ph = int(pw / aspect_ratio)
            else:
                pw = int(iw * crop_size)
                ph = int(iw * crop_size / aspect_ratio)
        else:
            if isinstance(crop_size, int):
                ph = crop_size
                pw = int(ph * aspect_ratio)
            else:
                ph = int(ih * crop_size)
                pw = int(ih * crop_size * aspect_ratio)

        return ph, pw

    def _get_patch_quad(self, theta, ph, pw):
        cos, sin = np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)
        hpx, hpy = 0.5 * pw, 0.5 * ph  # half patch size
        quad = np.array([[-hpx, -hpy], [hpx, -hpy], [hpx, hpy], [-hpx, hpy]], dtype=np.float32)
        rotation_m = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
        quad = np.matmul(quad, rotation_m)  # patch quadrilateral in relative coords

        return quad

    def _get_located_patch_quad(self, ih, iw, patch_quad_rel, bboxes=[]):
        image_poly = Polygon([[0, 0], [iw, 0], [iw, ih], [0, ih]])
        if self.min_image_overlap is not None:
            center_patch_poly = Polygon(
                np.array([0.5 * ih, 0.5 * iw], dtype=np.float32) + patch_quad_rel)
            max_available_overlap = (
                image_poly.intersection(center_patch_poly).area / center_patch_poly.area)
            min_image_overlap = min(self.min_image_overlap, max_available_overlap)
        else:
            min_image_overlap = None

        if self.min_bbox_count > 0:
            min_bbox_count = min(self.min_bbox_count, len(bboxes))
        else:
            min_bbox_count = 0

        cx_margin, cy_margin = np.sort(patch_quad_rel[:, 0])[2], np.sort(patch_quad_rel[:, 1])[2]

        found_randomly = False
        for trial_idx in range(self.max_random_trials):
            cx, cy = npr.uniform(cx_margin, iw - cx_margin), npr.uniform(cy_margin, ih - cy_margin)
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            patch_poly = Polygon(patch_quad)
            
            if min_image_overlap:
                image_overlap = patch_poly.intersection(image_poly).area / patch_poly.area
                # 이미지에서 벗어난 영역이 특정 비율보다 높으면 탈락
                if image_overlap < min_image_overlap:
                    continue

            if (self.min_bbox_count or not self.allow_partial_occurrence) and self.min_bbox_overlap:
                bbox_count = 0
                partial_occurrence = False
                
                for bbox in bboxes:
                    bbox_poly = Polygon(bbox)
                    if bbox_poly.area <= 0:
                        continue
                    
                    bbox_overlap = bbox_poly.intersection(patch_poly).area / bbox_poly.area
                    if bbox_overlap >= self.min_bbox_overlap:
                        bbox_count += 1
                    if (not self.allow_partial_occurrence and bbox_overlap > 0 and
                        bbox_overlap < self.min_bbox_overlap):
                        partial_occurrence = True
                        break
                
                # 부분적으로 나타나는 개체가 있으면 탈락
                if partial_occurrence:
                    continue
                # 온전히 포함하는 개체가 특정 개수 미만이면 탈락
                elif self.min_bbox_count and bbox_count < self.min_bbox_count:
                    continue

            found_randomly = True
            break

        if found_randomly:
            return patch_quad, trial_idx + 1
        else:
            return None, trial_idx + 1

    def crop_rotate_resize(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[]):
        """
        Args:
            image (ndarray): (H, W, C) shaped ndarray.
            masks (List[ndarray]): List of (H, W) shaped ndarray the same size as the image.
        """
        ih, iw = image.shape[:2]  # image height and width

        theta = self._get_theta()
        ph, pw = self._get_patch_size(ih, iw)

        patch_quad_rel = self._get_patch_quad(theta, ph, pw)

        if not self.random_translate:
            cx, cy = 0.5 * iw, 0.5 * ih
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            num_trials = 0
        else:
            patch_quad, num_trials = self._get_located_patch_quad(ih, iw, patch_quad_rel,
                                                                  bboxes=word_bboxes)

        vflip, hflip = self.vflip and npr.randint(2) > 0, self.hflip and npr.randint(2) > 0

        if self.resize_to is None:
            oh, ow = ih, iw
        elif self.keep_aspect_ratio:  # `resize_to`: Union[int, float]
            if self.resize_based_on == 'height' or self.resize_based_on == 'longest' and ih >= iw:
                oh = ih * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                ow = int(oh * iw / ih)
            else:
                ow = iw * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                oh = int(ow * ih / iw)
        elif isinstance(self.resize_to[0], float):  # `resize_to`: tuple[float, float]
            oh, ow = ih * self.resize_to[0], iw * self.resize_to[1]
        else:  # `resize_to`: tuple[int, int]
            oh, ow = self.resize_to

        if theta == 0 and (ph, pw) == (ih, iw) and (oh, ow) == (ih, iw) and not (hflip or vflip):
            M = None
            transformed = dict(image=image, word_bboxes=word_bboxes,
                               by_word_char_bboxes=by_word_char_bboxes, masks=masks)
        else:
            dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
            if patch_quad is not None:
                src = patch_quad
            else:
                if ow / oh >= iw / ih:
                    pad = int(ow * ih / oh) - iw
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[-off, 0], [iw + pad - off, 0], [iw + pad - off, ih], [-off, ih]],
                        dtype=np.float32)
                else:
                    pad = int(oh * iw / ow) - ih
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[0, -off], [iw, -off], [iw, ih + pad - off], [0, ih + pad - off]],
                        dtype=np.float32)

            if hflip:
                src = src[[1, 0, 3, 2]]
            if vflip:
                src = src[[3, 2, 1, 0]]

            M = cv2.getPerspectiveTransform(src, dst)
            transformed = transform_by_matrix(M, image=image, oh=oh, ow=ow, word_bboxes=word_bboxes,
                                              by_word_char_bboxes=by_word_char_bboxes, masks=masks)

        found_randomly = self.random_translate and patch_quad is not None

        return dict(found_randomly=found_randomly, num_trials=num_trials, matrix=M, **transformed)


class ComposedTransformation:
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        random_translate=False, min_image_overlap=0, min_bbox_overlap=0, min_bbox_count=0,
        allow_partial_occurrence=True,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100,
        brightness=0, contrast=0, saturation=0, hue=0,
        normalize=False, mean=None, std=None, to_tensor=False
    ):
        self.geo_transform_fn = GeoTransformation(
            rotate_anchors=rotate_anchors, rotate_range=rotate_range,
            crop_aspect_ratio=crop_aspect_ratio, crop_size=crop_size, crop_size_by=crop_size_by,
            hflip=hflip, vflip=vflip, random_translate=random_translate,
            min_image_overlap=min_image_overlap, min_bbox_overlap=min_bbox_overlap,
            min_bbox_count=min_bbox_count, allow_partial_occurrence=allow_partial_occurrence,
            resize_to=resize_to, keep_aspect_ratio=keep_aspect_ratio,
            resize_based_on=resize_based_on, max_random_trials=max_random_trials)

        alb_fns = []
        if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
            alb_fns.append(A.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1))

        if normalize:
            kwargs = dict()
            if mean is not None:
                kwargs['mean'] = mean
            if std is not None:
                kwargs['std'] = std
            alb_fns.append(A.Normalize(**kwargs))

        if to_tensor:
            alb_fns.append(ToTensorV2())

        self.alb_transform_fn = A.Compose(alb_fns)

    def __call__(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[], height_pad_to=None,
                 width_pad_to=None):
        # TODO Seems that normalization should be performed before padding.

        geo_result = self.geo_transform_fn(image, word_bboxes=word_bboxes,
                                           by_word_char_bboxes=by_word_char_bboxes, masks=masks)

        if height_pad_to is not None or width_pad_to is not None:
            min_height = height_pad_to or geo_result['image'].shape[0]
            min_width = width_pad_to or geo_result['image'].shape[1]
            alb_transform_fn = A.Compose([
                A.PadIfNeeded(min_height=min_height, min_width=min_width,
                              border_mode=cv2.BORDER_CONSTANT,
                              position=A.PadIfNeeded.PositionType.TOP_LEFT),
                self.alb_transform_fn])
        else:
            alb_transform_fn = self.alb_transform_fn
        final_result = alb_transform_fn(image=geo_result['image'])
        del geo_result['image']

        return dict(image=final_result['image'], **geo_result)
