import random
import torch
import numpy as np

def _get_random_crop_indices(crop_region, crop_size):
    '''
    crop_region: (strat_y, end_y, start_x, end_x)
    crop_size: (y, x)
    '''
    region_size = (crop_region[1] - crop_region[0], crop_region[3] - crop_region[2])
    if region_size[0] < crop_size[0] or region_size[1] < crop_size[1]:
        print(region_size, crop_size)
    assert region_size[0] >= crop_size[0] and region_size[1] >= crop_size[1]
    if region_size[0] == crop_size[0]:
        start_y = crop_region[0]
    else:
        start_y = random.choice(range(crop_region[0], crop_region[1] - crop_size[0]))
    if region_size[1] == crop_size[1]:
        start_x = crop_region[2]
    else:
        start_x = random.choice(range(crop_region[2], crop_region[3] - crop_size[1]))
    return start_y, start_y + crop_size[0], start_x, start_x + crop_size[1]

def _get_adaptive_crop_indices(crop_region, crop_size, num_candidate, dist_map, min_diff=False):
    candidates = [_get_random_crop_indices(crop_region, crop_size) for _ in range(num_candidate)]
    max_choice = candidates[0]
    min_choice = candidates[0]
    max_dist = 0
    min_dist = np.infty 
    with torch.no_grad():
        for c in candidates:
            start_y, end_y, start_x, end_x = c
            dist = torch.sum(dist_map[start_y: end_y, start_x: end_x])
            if dist > max_dist:
                max_dist = dist
                max_choice = c
            if dist < min_dist:
                min_dist = dist
                min_choice = c
    if min_diff:
        return min_choice
    else:
        return max_choice

def get_split_list(divisor, dividend):
    split_list = [dividend // divisor for _ in range(divisor - 1)]
    split_list.append(dividend - (dividend // divisor) * (divisor - 1))
    return split_list

def random_sampler(pic_size, crop_dict):
    crop_region = (0, pic_size[0], 0, pic_size[1])
    crop_res_dict = {}
    for k, v in crop_dict.items():
        crop_size = (int(k), int(k))
        crop_res_dict[k] = [_get_random_crop_indices(crop_region, crop_size) for _ in range(v)]
    return crop_res_dict

def region_sampler(crop_region, crop_dict):
    crop_res_dict = {}
    for k, v in crop_dict.items():
        crop_size = (int(k), int(k))
        crop_res_dict[k] = [_get_random_crop_indices(crop_region, crop_size) for _ in range(v)]
    return crop_res_dict

def adaptive_sampler(pic_size, crop_dict, num_candidate_dict, dist_map, min_diff=False):
    crop_region = (0, pic_size[0], 0, pic_size[1])
    crop_res_dict = {}
    for k, v in crop_dict.items():
        crop_size = (int(k), int(k))
        crop_res_dict[k] = [_get_adaptive_crop_indices(crop_region, crop_size, num_candidate_dict[k], dist_map, min_diff) for _ in range(v)]
    return crop_res_dict

# TODO more flexible
def pyramid_sampler(pic_size, crop_dict):
    crop_res_dict = {}
    sorted_key = list(crop_dict.keys())
    sorted_key.sort(key=lambda x: int(x), reverse=True)
    k = sorted_key[0]
    crop_size = (int(k), int(k))
    crop_region = (0, pic_size[0], 0, pic_size[1])
    crop_res_dict[k] = [_get_random_crop_indices(crop_region, crop_size) for _ in range(crop_dict[k])]

    for i in range(1, len(sorted_key)):
        crop_res_dict[sorted_key[i]] = []
        afore_num = crop_dict[sorted_key[i-1]]
        new_num = crop_dict[sorted_key[i]]
        split_list = get_split_list(afore_num, new_num)
        crop_size = (int(sorted_key[i]), int(sorted_key[i]))
        for j in range(len(split_list)):
            crop_region = crop_res_dict[sorted_key[i-1]][j]
            crop_res_dict[sorted_key[i]].extend([_get_random_crop_indices(crop_region, crop_size) for _ in range(split_list[j])])

    return crop_res_dict

