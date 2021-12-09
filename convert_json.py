import os.path as osp
import json
import os
from tqdm import tqdm
import copy

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

add_data_dir = os.environ.get('data', '/opt/ml/input/data/Revised')
# 여기 부분 가지고 계신 폴더구성에 맞추어 수정해주시구요

with open(osp.join(add_data_dir, 'annotation.json'.format('annotation')), 'r') as f:
# 여기 부분도 가지고 계신 파일명에 맞추어서 수정해주시길 요청드립니다.
    anno = json.load(f)
# illegibility는 전부 단어 이므로 false
anno = anno['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0
count_none_anno = 0

for img_name, img_info in tqdm(anno.items()) :
    if img_info['words'] == {} :
        del(anno_temp[img_name])
        count_none_anno += 1
        continue
    for obj, obj_info in img_info['words'].items() :
        # illegibility는 전부 단어 이므로 false
        anno_temp[img_name]['words'][obj]['illegibility'] = False
        if len(img_info['words'][obj]['points']) == 4 :
            count_normal += 1
            continue
            
        elif len(img_info['words'][obj]['points']) < 4 :
            del(anno_temp[img_name]['words'][obj])
        # 폴리곤 수정시에는 여기 부분을 수정해주시면 됩니다!!
        # 다음 예제는 polygon이 넘칠 경우 해당 폴리곤을 illegibility를 삭제처리
            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
        else :
            del(anno_temp[img_name]['words'][obj])
            # 이부분은 resize시 에러가 나서 우선 참고로만 적어놓고 주석처리했습니다.
            # anno_temp[img_name]['words'][obj]['illegibility'] = True
            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
            count += 1
            
print(f'normal polygon count : {count_normal}')
print(f'deleted {count} over polygon')
print(f'{count_none_anno} images deleted')

anno = {'images': anno_temp}

ufo_dir = osp.join('../input/data/Revised', 'ufo')
maybe_mkdir(ufo_dir)
with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
    json.dump(anno, f, indent=4)