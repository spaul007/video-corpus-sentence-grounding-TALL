" Some useful functions "

import numpy as np
from six.moves import xrange
import time
import pickle
import operator
import torch
import os
import sys


def confirm_file(file_path):
    # file_path: path of the file that need to be checked
    assert os.path.isfile(file_path), file_path +' does not exit' 
    
    
def read_pkl_file(filepath):
    assert os.path.isfile(filepath), filepath + ' does not exist'
    pickledic = None
    try:
        if sys.version_info[0] < 3:
            # logger.info("\n\n[info pickledic_utils] PYTHON 3")
            with open(filepath,'rb') as infile:
                pickledic = pickle.load(infile)
        else:
            # with open(filepath, "rb") as infile:
            #     pickledic = pickle.load(infile, encoding='latin1')
            # logger.info("\n\n[info pickledic_utils] PYTHON>=3 ")
            with open(filepath, 'rb') as infile:
                u = pickle._Unpickler(infile)
                u.encoding = 'latin1'
                pickledic = u.load()
                # logger.info("pickledic: {}".format(pickledic))
    except:
        print('pickle loading did not work') 
    return pickledic
    
    

def calculate_reward_batch_withstop(Previou_IoU, current_IoU, t):
    batch_size = len(Previou_IoU)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        if current_IoU[i] > Previou_IoU[i] and Previou_IoU[i]>=0:
            reward[i] = 1 -0.001*t
        elif current_IoU[i] <= Previou_IoU[i] and current_IoU[i]>=0:
            reward[i] = -0.001*t
        else:
            reward[i] = -1 -0.001*t
    return reward


def calculate_reward(Previou_IoU, current_IoU, t):
    if current_IoU > Previou_IoU and Previou_IoU>=0:
        reward = 1-0.001*t
    elif current_IoU <= Previou_IoU and current_IoU>=0:
        reward = -0.001*t
    else:
        reward = -1-0.001*t

    return reward

def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        # if inter[1] < inter[0]:
        #     iou = 0
        # else:
        iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
        iou_batch[i] = iou
    return iou_batch

def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL


def nms_temporal(x1,x2,s,overlap):
    # x1: starting times
    # x2: ending times
    # s: allignment score
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    #union = map(operator.sub, x2, x1) # union = x2-x1
    # editing for python36
    union = list(map(operator.sub, x2, x1)) # union = x2-x1 # getting duration

    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index
    
    # for each temporal range with highest allignment score, calculate the overlap region
    # xx1: starting time of overlap portion with the clip of highest allignment score 
    # xx1: ending time of overlap portion with the clip of highest allignment score
    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]  # start time taking the rightmost one!
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]] # end time taking the left most one! 
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)] # start time end time difference
        # 
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)] # 
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick


def compute_IoU_recall_top_n_forreg_rl(top_n, iou_thresh, sentence_image_reg_mat, sclips):
    correct_num = 0.0
    for k in range(sentence_image_reg_mat.shape[0]):
        gt = sclips[k]
        # print(gt)
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])

        pred_start = sentence_image_reg_mat[k, 0]
        pred_end = sentence_image_reg_mat[k, 1]
        iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
        if iou>=iou_thresh:
            correct_num+=1

    return correct_num

def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    
    # for each sentence
    for k in range(sentence_image_mat.shape[0]):
        # take the ground truth info from sclips
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        
        # snetence_image_mat: [sentence,clip]
        
        # sim_v : list of all clips for a single sentence
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        
        # non maximum suppression in temporal dimension
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num

def compute_IoU_recall_top_n_forreg_testset(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips, gt_idx):
    # top_n : recall@top_n
    # sentence_image_mat: [sentence,clips] allignment score
    # sentence_image_reg_mat: [sentence,clips,2] temporal offsets start and end time offsets
    # sclips : list of sentence clips
    # iclips : list of all videos, each video list contains all the clips
    # gt_idx : actual index of the clip that contains the query sentence
    
    
    correct_num = 0.0
    
    # for each sentence (there is only one sentence)
    for k in range(sentence_image_mat.shape[0]):
        # take the ground truth info from sclips
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        
        # snetence_image_mat: [sentence,clip]
        
        # sim_v : list of all clips for a single sentence
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        
        # non maximum suppression in temporal dimension
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num

def compute_IoU_recall_top_n_forreg_corpus(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips_all, v_idx):
    # top_n : recall@top_n
    # sentence_image_mat: [sentences,clips] allignment score
    # sentence_image_reg_mat: [sentences,clips,2] temporal offsets start and end time offsets
    # s_clips : list of sentence clips
    # iclips_all : list of all videos, each video list contains all the clips
    # v_idx : actual index of the video that contains the query sentence
    correct_num = 0.0
    picks_all = []
    score_idx_pair = [] 
           
    # for each sentence
    for k in range(sentence_image_mat.shape[0]):
        # take the ground truth info from sclips
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        # need to know all gt indexes
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        
        iclip_idx = 0
        for cur_vid, iclips in enumerate(iclips_all):
            n_iclip = len(iclips) # number of clips in that video
            
            if v_idx == cur_vid:
                gt_idxes = [i for i in range(iclip_idx,iclip_idx+n_iclip)]
            
            # iclip_idx:(iclip_idx+n_iclip) :current index to current index + number of clips
            sim_v = [v for v in sentence_image_mat[k,iclip_idx:(iclip_idx+n_iclip)]]
            starts = [s for s in sentence_image_reg_mat[k,iclip_idx:(iclip_idx+n_iclip),0]]
            ends = [e for e in sentence_image_reg_mat[k,iclip_idx:(iclip_idx+n_iclip),1]]
            # non maximum suppression in temporal dimension
            # picks only contains the index
            
            picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
            # trying to check what happens if iou-.05 not used
            # picks = nms_temporal(starts,ends, sim_v, iou_thresh)
            
            # iou threshold has an impact on nms_temporal
            for idx in  picks:
                picks_all.append(idx+iclip_idx) # all picked idx in sentence_image_mat 
                score_idx_pair.append((sentence_image_mat[k,idx+iclip_idx],idx+iclip_idx))
            iclip_idx += n_iclip
            
        picks_all_sorted = [i[1] for i in sorted(score_idx_pair, key=lambda x:x[0])] # sort and get index
    
        
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks_all_sorted): picks_all_sorted=picks_all_sorted[0:top_n]
        
        for index in picks_all_sorted:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if (iou>=iou_thresh and index in gt_idxes):
                correct_num+=1
                break
    return correct_num


