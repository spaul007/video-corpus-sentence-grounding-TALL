" Dataloader of charades-STA dataset for Supervised Learning based methods"

import torch
import torch.utils.data
import os
import pickle
import numpy as np
from utils import *



def create_testset():
        
    test_swin_txt_path = "./Dataset/Charades/ref_info/charades_sta_test_swin_props_num_36364.txt"
    clip_sentence_pairs = read_pkl_file("./Dataset/Charades/ref_info/charades_sta_test_semantic_sentence_VP_sub_obj.pkl")
    clip_sentence_test = {}
    
    sliding_clip_names = []
    with open(test_swin_txt_path) as f:
        for l in f:
            sliding_clip_names.append(l.rstrip().replace(" ", "_"))
                
    # clip_sentence_pair: dict
    # sliding_clip_names: list
        
    # movie is a key of the clip_sentence_pair dict
    for movie in clip_sentence_pairs:
        clip_sentence_test[movie]={}
        # clip_sentence_pair[movie_name] is a dict, clip is a key
        for clip in clip_sentence_pairs[movie]:   
            # clip is the clip_name, gives groundtruth start and end time
            gt_start = int(clip.split('_')[1])
            gt_end = int(clip.split('_')[2])
                
            for coarse_clip_info in sliding_clip_names: 
                if coarse_clip_info.split('_')[0] == movie:
                    # check IoU, nIoL
                    cur_start = int(coarse_clip_info.split('_')[1])
                    cur_end = int(coarse_clip_info.split('_')[2])
                    # check IoU
                    IoU = 0
                    nIoL = 1
                    IoU = calculate_IoU((cur_start,cur_end),(gt_start,gt_end))
                    if IoU > 0.5 and nIoL< .2:
                        nIoL = calculate_nIoL((gt_start,gt_end),(cur_start,cur_end))
                        if IoU >.5 :
                            new_clip_name = coarse_clip_info.split('_')[0] + '_' + coarse_clip_info.split('_')[1] + '_' + coarse_clip_info.split('_')[2]
                            clip_sentence_test[movie][new_clip_name]=[]
                            # for different senteces of same gt time segment, store the sent_skip_thought_vec
                            for elem in clip_sentence_pairs[movie][clip]:
                                clip_sentence_test[movie][new_clip_name].append({'sent_skip_thought_vec': elem['sent_skip_thought_vec']})
                    
    
    pickle_file_name = 'aligned_testset.pkl'
    pickle_path = './Dataset/Charades/ref_info'
    pickle_file = os.path.join(pickle_path, pickle_file_name)
    
    with open(pickle_file,'wb') as F:
        pickle.dump(clip_sentence_test,F)
        
    return clip_sentence_test
       
if __name__ == '__main__':
    aa = create_testset()

        
        