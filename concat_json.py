import os.path as osp
import json
import os
from tqdm import tqdm
import copy

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

add_data_dir_1 = os.environ.get('data', '/opt/ml/input/data/Revised/ufo')
# 여기 부분 가지고 계신 폴더구성에 맞추어 수정해주시구요

with open(osp.join(add_data_dir_1, 'train.json'.format('annotation')), 'r') as f:
# 여기 부분도 가지고 계신 파일명에 맞추어서 수정해주시길 요청드립니다.
    anno_1 = json.load(f)
# illegibility는 전부 단어 이므로 false
anno_1 = anno_1['images']

anno_temp_1 = copy.deepcopy(anno_1)
print(type(anno_temp_1))
print(len(anno_temp_1))

add_data_dir_1 = os.environ.get('data', '/opt/ml/input/data/ICDAR17_Korean/ufo')

with open(osp.join(add_data_dir_1, 'train.json'.format('annotation')), 'r') as f:
    anno_2 = json.load(f)

anno_2 = anno_2['images']

anno_temp_2 = copy.deepcopy(anno_2)
print(type(anno_temp_2))
print(len(anno_temp_2))

anno_temp_1.update(anno_temp_2)
print(len(anno_temp_1))

anno = {'images': anno_temp_1}

with open('/opt/ml/input/data/Revised/ufo/train_total.json', 'w') as f:
    json.dump(anno, f, indent=4)