" Train and test file for Supervised Learning based methods (TALL & MAC) for Charades-STA dataset \
TALL: Temporal Activity Localization via Language Query(http://openaccess.thecvf.com/content_ICCV_2017/papers/Gao_TALL_Temporal_Activity_ICCV_2017_paper.pdf) \
MAC: Mining Activity Concepts for Language-based Temporal Localization (https://arxiv.org/pdf/1811.08925.pdf) "

from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']="2"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np

import argparse
from utils import *
import random
from torch.autograd import Variable
from dataloader_charades_SL import Charades_Train_dataset, Charades_Test_dataset
from model_TALL import TALL
from model_MAC import MAC


parser = argparse.ArgumentParser(description='Video Grounding of PyTorch')
parser.add_argument('--model', type=str, default='TALL', help='model type') # TALL, MAC
parser.add_argument('--dataset', type=str, default='Charades', help='dataset type')
parser.add_argument('--batch_size', default=56, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

# new
parser.add_argument('--weight_path', type=str, default='./Charades_TALL/best_R5_IOU5_model.t7', help='path to model weights') # TALL, MAC
parser.add_argument('--test_type', type=str, default='single_video', help='test on single or all video') # single_video, all_video

opt = parser.parse_args()

path = os.path.join(opt.dataset + '_' + opt.model)

train_dataset = Charades_Train_dataset()
test_dataset = Charades_Test_dataset()

num_train_batches = int(len(train_dataset)/opt.batch_size)
print ("num_train_batches:", num_train_batches)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4)


# Model
if opt.model == 'TALL':
    net = TALL().cuda()
elif opt.model == 'MAC':
    net = MAC().cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

setup_seed(0)
best_R1_IOU5 = 0
best_R5_IOU5 = 0
best_R1_IOU5_epoch = 0
best_R5_IOU5_epoch = 0

# Training
def train(epoch):
    net.train()
    train_loss = 0

    # i am guesing it is only taking clip sentence pair, not taking any clip that has no caption
    for batch_idx, (images, sentences, offsets, softmax_center_clips, VP_spacys) in enumerate(trainloader):
        images, sentences, offsets, softmax_center_clips, VP_spacys = images.cuda(), sentences.cuda(), offsets.cuda(), softmax_center_clips.cuda(), VP_spacys.cuda()

        # network forward
        if opt.model == 'TALL':
            outputs = net(images, sentences)
        elif opt.model == 'MAC':
            outputs = net(images, sentences, softmax_center_clips, VP_spacys)

        # compute alignment and regression loss
        sim_score_mat = outputs[0]
        p_reg_mat = outputs[1]
        l_reg_mat = outputs[2]
        
        # loss cls, not considering iou
        # outputs is a tensor: size of the 2nd dimension(axis=1): ?
        # what is input? should be the number of clips, not sure
        input_size = outputs.size(1)
        I = torch.eye(input_size).cuda()  # [sentence,sentence]??? identity matrix
        I_2 = -2 * I  # why?
        all1 = torch.ones(input_size, input_size).cuda() # [sentence,sentence]

        mask_mat = I_2 + all1  # 56,56 #sudipta: why 56? is this the size of the batch?

        #               | -1  1   1...   |
        #   mask_mat =  | 1  -1   1...   |
        #               | 1   1  -1 ...  |

        alpha = 1.0 / input_size  # 
        lambda_regression = 0.01
        batch_para_mat = alpha * all1 # making it 1/56 for each eement
        para_mat = I + batch_para_mat# making the diagonal element 55/56, why?

        loss_mat = torch.log(all1 + torch.exp(mask_mat*sim_score_mat)) #negative similarity for the actual sentence clip pair, according to the loss function
        loss_mat = loss_mat*para_mat # for bringing everthing in same scale -> positive pair already 55/56 and negative pairs (1/56)*55
        loss_align = loss_mat.mean() # then average out

        # regression loss
        # linear algebra to take diagonal element
        # catenating in dimension 1
        # actual offset is known, subtracting and taking abs value, and then taking mean
        l_reg_diag = torch.mm(l_reg_mat*I, torch.ones(input_size, 1).cuda())
        p_reg_diag = torch.mm(p_reg_mat*I, torch.ones(input_size, 1).cuda())
        offset_pred = torch.cat([p_reg_diag, l_reg_diag], 1)
        loss_reg = torch.abs(offset_pred - offsets).mean() # L1 loss

        loss= lambda_regression*loss_reg + loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        print('Epoch: %d | Step: %d | Loss: %.3f | loss_align: %.3f | loss_reg: %.3f' % (epoch, batch_idx, train_loss / (batch_idx + 1), loss_align, loss_reg))


def test_corpus_only_pairs(weight_path):
    
    iclips = [b[0] for b in movie_clip_featmaps]
    sclips = [b[0] for b in movie_clip_sentences]
       
    # loading model
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['net'])  
    net.eval()
    
    # threshold
    IoU_thresh = [0.1, 0.3, 0.5, 0.7]    
    all_correct_num_100 = [0.0] * 5 # why 5?
    
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names) # number of the movies, not number of clips
    iclips_all = [] # contains all clips of all videos # list with 
    
    # store all the fetures in a list
    clip_features = []
    for clip in test_dataset.clip_sentence_test_list:
        clip_features.append(test_dataset.load_video_feature(clip['clip_name']))         
    
    print('clip features loaded')
    
    # set how you want to take the features
    testclip_num = len(clip_features) # 10554
    clip_increment = 200  # taking 200, because 500 resulting in memory error
                    
    # from each test clip, 
    for idx1, clip in enumerate(test_dataset.clip_sentence_test_list):
        # to store the allignment score
        sentence_image_mat = np.zeros([1, testclip_num])
        sentence_image_reg_mat = np.zeros([1, testclip_num,2])
        
        start_frame = np.zeros([1,testclip_num]) # offset is based on frame number probably
        end_frame = np.zeros([1,testclip_num])
        
        # get sentence vector first
        sent_vec = test_dataset.load_sentence_feature(clip['clip_name'])
        sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800  # why this size?
        
        for idx2 in range(0,testclip_num,clip_increment): # idx2: 0,200,400,...10400 
    
            featmap = np.zeros((clip_features[0].shape[0],clip_increment)) #(4096*3,200)
            
            # assigning values in the featmap numpy from clip_features list
            if (idx2+clip_increment) <= testclip_num:
                for idx3 in range(idx2,idx2+clip_increment): # range(0,200), range(200,400)... range(10200,10400)
                    featmap[:,idx3-idx2] = clip_features[idx3]
                    start_frame[0,idx3] = float(test_dataset.clip_sentence_test_list[idx3]['clip_name'].split('_')[1])
                    end_frame[0,idx3] = float(test_dataset.clip_sentence_test_list[idx3]['clip_name'].split('_')[2])
                    
            else:
                featmap = np.zeros((clip_features[0].shape[0],testclip_num-idx2)) #(12288?,10554-10400)
                for idx3 in range(idx2,testclip_num): # range(10400,10554)
                    featmap[:,idx3-idx2] = clip_features[idx3]
                    start_frame[0,idx3] = float(test_dataset.clip_sentence_test_list[idx3]['clip_name'].split('_')[1])
                    end_frame[0,idx3] = float(test_dataset.clip_sentence_test_list[idx3]['clip_name'].split('_')[2])
            
            # transposing and sending to gpu
            featmap = np.transpose(featmap) # now the shape: (elem,feat)
            featmap = torch.from_numpy(featmap).float().cuda()
            
            # repeating and gpu (i need to repeat because in the actual test function they used only one instance) 
            sent_vec = np.repeat(sent_vec,featmap.shape[0],axis=0)
            sent_vec = torch.from_numpy(sent_vec).float().cuda()
            
            # network forward
            if opt.model == 'TALL':
                with torch.no_grad():
                    outputs = net(featmap, sent_vec)
            elif opt.model == 'MAC':
                assert opt.model == 'TALL', 'model not TALL, MAC used'
            
            outputs = outputs.squeeze(1).squeeze(1)
            outputs = outputs.cpu().numpy()
            # output[0]: confidence score
            # output[1]: offset on the starting time? 
            # output[2]: ofset on the right side?
            
            if (idx2+clip_increment) <= testclip_num:
                sentence_image_mat[0,idx2:idx2+clip_increment] = outputs[0,0,:]
                reg_end = end_frame[0,idx2:idx2+clip_increment] + outputs[2,0,:]
                reg_start = start_frame[0,idx2:idx2+clip_increment] + outputs[1,0,:]
    
                sentence_image_reg_mat[0, idx2:idx2+clip_increment, 0] = reg_start
                sentence_image_reg_mat[0, idx2:idx2+clip_increment, 1] = reg_end
            else:
                sentence_image_mat[0,idx2:testclip_num] = outputs[0,0,:]
                reg_end = end_frame[0,idx2:testclip_num] + outputs[2,0,:]
                reg_start = start_frame[0,idx2:testclip_num] + outputs[1,0,:]
    
                sentence_image_reg_mat[0, idx2:testclip_num, 0] = reg_start
                sentence_image_reg_mat[0, idx2:testclip_num, 1] = reg_end
                
        
        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            # ********modified for corpus
            correct_num_100 = compute_IoU_recall_top_n_forreg_corpus(100, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips_all, idx1) # actual index of video that contains the sentence
            print(movie_name_1 + " IoU=" + str(IoU) + ", R@100: " + str(correct_num_100 / len(sclips)))
            
            all_correct_num_100[k] += correct_num_100
            #all_correct_num_5[k] += correct_num_5
            #all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
        
    # for different threshold, print all testset output resullt   
    assert os.path.isdir('Charades_TALL'), 'Charades_TALL folder doesnot exist'
    test_result_output=open(os.path.join('Charades_TALL', "test_results_corpus.txt"), "w")
   
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(all_correct_num_100[k] / all_retrievd))

        test_result_output.write("IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(
            all_correct_num_100[k] / all_retrievd))
                
            
            
        
   
    
    
    # first loop: for each movie (to get the sentences)
    for idx1, movie_name_1 in enumerate(test_dataset.movie_names):
        n_clip = 0 # to trace where to put scores
        print("sentence belongs to movie no.: %d/%d" %(idx1, all_number)) #(current movie/all movie) number
        # getting the sentences of the movie
        _ , movie_clip_sentences = test_dataset.load_movie_slidingclip(movie_name_1, 16)
        #print("number of sentences: " + str(len(movie_clip_sentences)))
        sclips = [b[0] for b in movie_clip_sentences]
        
        # 2nd loop: looping over the movie names (to get the clips)
        for idx2, movie_name_2 in enumerate(test_dataset.movie_names):
            
            movie_clip_featmaps, _ = test_dataset.load_movie_slidingclip(movie_name_2, 16)
            #print("number of clips: " + str(len(movie_clip_featmaps)))  # candidate clips)
            if idx2%50 == 0:
                print("video clip belongs to movie no.: %d/%d" %(idx2, all_number))
                
            # need to catenate the matrices
            if idx2 == 0:
                sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
                sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
            else:
                sentence_image_mat = np.concatenate((sentence_image_mat,np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])),axis=1)
                sentence_image_reg_mat = np.concatenate((sentence_image_reg_mat,np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps),2])),axis=1)
                
            # for each sentence
            for k in range(len(movie_clip_sentences)):
            
                # get the sent vec
                sent_vec = movie_clip_sentences[k][1]
                sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800  # why this size?
                sent_vec = torch.from_numpy(sent_vec).cuda()
    
                VP_spacy_vec = movie_clip_sentences[k][2]
                VP_spacy_vec = np.reshape(VP_spacy_vec, [1, VP_spacy_vec.shape[0]])
                VP_spacy_vec = torch.from_numpy(VP_spacy_vec).float().cuda()
                
                for t in range(len(movie_clip_featmaps)):
                    # for each sentence, slide through all the clips
                    featmap = movie_clip_featmaps[t][1] #12288 ??
                    visual_clip_name = movie_clip_featmaps[t][0]
                    softmax_ = movie_clip_featmaps[t][2]
    
                    start = float(visual_clip_name.split("_")[1])
                    end = float(visual_clip_name.split("_")[2].split("_")[0])
                    conf_score = float(visual_clip_name.split("_")[7])
    
                    featmap = np.reshape(featmap, [1, featmap.shape[0]])
                    featmap = torch.from_numpy(featmap).cuda()
    
                    softmax_ = np.reshape(softmax_, [1, softmax_.shape[0]])
                    softmax_ = torch.from_numpy(softmax_).cuda()
    
                    # network forward
                    if opt.model == 'TALL':
                        outputs = net(featmap, sent_vec)
                    elif opt.model == 'MAC':
                        outputs = net(featmap, sent_vec, softmax_, VP_spacy_vec)
    
                    outputs = outputs.squeeze(1).squeeze(1)
                    # output[0]: confidence score
                    # output[1]: offset on the starting time? 
                    # output[2]: ofset on the right side?
    
                    if opt.model == 'TALL':
                        sentence_image_mat[k, t + n_clip] = outputs[0]
                    elif opt.model == 'MAC':
                        sigmoid_output0 = 1 / float(1 + torch.exp(-outputs[0]))
                        sentence_image_mat[k, t + n_clip] = sigmoid_output0 * conf_score
    
                    # sentence_image_mat[k, t] = expit(outputs[0]) * conf_score
                    reg_end = end + outputs[2]
                    reg_start = start + outputs[1]
    
                    sentence_image_reg_mat[k, t + n_clip, 0] = reg_start
                    sentence_image_reg_mat[k, t + n_clip, 1] = reg_end
            
            n_clip += len(movie_clip_featmaps)               
            iclips = [b[0] for b in movie_clip_featmaps]
            iclips_all.append(iclips)

            
        
        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            # ********modified for corpus
            correct_num_100 = compute_IoU_recall_top_n_forreg_corpus(100, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips_all, idx1) # actual index of video that contains the sentence
            print(movie_name_1 + " IoU=" + str(IoU) + ", R@100: " + str(correct_num_100 / len(sclips)))
            
            all_correct_num_100[k] += correct_num_100
            #all_correct_num_5[k] += correct_num_5
            #all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
        
    # for different threshold, print all testset output resullt   
    assert os.path.isdir('Charades_TALL'), 'Charades_TALL folder doesnot exist'
    test_result_output=open(os.path.join('Charades_TALL', "test_results_corpus.txt"), "w")
   
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(all_correct_num_100[k] / all_retrievd))

        test_result_output.write("IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(
            all_correct_num_100[k] / all_retrievd))
        
        
def test(weight_path):
    
    """ This function loads a trained model and perform temporal localization on a video (not corpus)"""
    
    # where am i providing the model weight paths
    # need to load the model weights
    
    # saving and loading general chekpoint for inference of pytorch tutorial
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['net'])  
    
    #for param_tensor in net.state_dict():
    #    print(param_tensor)
    
    net.eval()
    
    # threshold
    IoU_thresh = [0.1, 0.3, 0.5, 0.7]    
    # for R@10?
    all_correct_num_10 = [0.0] * 5 # why 5?
    # for R@5?
    all_correct_num_5 = [0.0] * 5
    # for R@1?
    all_correct_num_1 = [0.0] * 5
    # not sure # probably ground truth number of sentence that means number of pairs to be retrieved
    all_retrievd = 0.0
    # number of all movies
    all_number = len(test_dataset.movie_names)
    idx = 0
    
    # for per movie
    for movie_name in test_dataset.movie_names:
        idx += 1
        print("%d/%d" %(idx, all_number)) #(current movie/all movie) number

        movie_clip_featmaps, movie_clip_sentences = test_dataset.load_movie_slidingclip(movie_name, 16)
        # movie clip featmap contains feature of all movie
        # movie_clips_sentences contain all sentences of a movie
        
        print("sentences: " + str(len(movie_clip_sentences)))
        print("clips: " + str(len(movie_clip_featmaps)))  # candidate clips)
        
        # sentence_image_mat: [sentence,clip] confidance score
        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        # sentence_image_reg_mat: [sentence,clip,2] start and end time 
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        
        # for each sentence
        for k in range(len(movie_clip_sentences)):
            
            # get the sent vec
            sent_vec = movie_clip_sentences[k][1]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800  # why this size?
            sent_vec = torch.from_numpy(sent_vec).cuda()

            VP_spacy_vec = movie_clip_sentences[k][2]
            VP_spacy_vec = np.reshape(VP_spacy_vec, [1, VP_spacy_vec.shape[0]])
            VP_spacy_vec = torch.from_numpy(VP_spacy_vec).float().cuda()
            
            for t in range(len(movie_clip_featmaps)):
                # for each sentence, slide through all the clips
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]
                softmax_ = movie_clip_featmaps[t][2]

                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split("_")[0])
                conf_score = float(visual_clip_name.split("_")[7])

                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                featmap = torch.from_numpy(featmap).cuda()

                softmax_ = np.reshape(softmax_, [1, softmax_.shape[0]])
                softmax_ = torch.from_numpy(softmax_).cuda()

                # network forward
                if opt.model == 'TALL':
                    outputs = net(featmap, sent_vec)
                elif opt.model == 'MAC':
                    outputs = net(featmap, sent_vec, softmax_, VP_spacy_vec)

                outputs = outputs.squeeze(1).squeeze(1)
                # output[0]: confidence score
                # output[1]: offset on the starting time? 
                # output[2]: ofset on the right side?

                if opt.model == 'TALL':
                    sentence_image_mat[k, t] = outputs[0]
                elif opt.model == 'MAC':
                    sigmoid_output0 = 1 / float(1 + torch.exp(-outputs[0]))
                    sentence_image_mat[k, t] = sigmoid_output0 * conf_score

                # sentence_image_mat[k, t] = expit(outputs[0]) * conf_score
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

        # taking all image clip and sentence clip in a list?
        # it contains the ground truth? maybe sclips onlu=y contains the ground truth
        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]


        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(
                IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
                correct_num_1 / len(sclips)))

            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
        
    # for different threshold, print all testset output resullt    
    assert os.path.isdir('Charades_TALL'), 'Charades_TALL folder doesnot exist'
    test_result_output=open(os.path.join('Charades_TALL', "test_results_video.txt"), "w")
    
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))

        test_result_output.write("IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(
            all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")


def test_corpus(weight_path):
    # try to create a logger
   
    # check test function for proper comments
    # loading model
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['net'])  
    net.eval()
    
    # threshold
    IoU_thresh = [0.1, 0.3, 0.5, 0.7]    
    all_correct_num_100 = [0.0] * 5 # why 5?
    #all_correct_num_10 = [0.0] * 5 # why 5?
    #all_correct_num_5 = [0.0] * 5
    #all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names)
    iclips_all = [] # contains all clips of all videos # list with 
    
    # first loop: for each movie (to get the sentences)
    for idx1, movie_name_1 in enumerate(test_dataset.movie_names):
        n_clip = 0 # to trace where to put scores
        print("sentence belongs to movie no.: %d/%d" %(idx1, all_number)) #(current movie/all movie) number
        # getting the sentences of the movie
        _ , movie_clip_sentences = test_dataset.load_movie_slidingclip(movie_name_1, 16)
        #print("number of sentences: " + str(len(movie_clip_sentences)))
        sclips = [b[0] for b in movie_clip_sentences]
        
        # 2nd loop: looping over the movie names (to get the clips)
        for idx2, movie_name_2 in enumerate(test_dataset.movie_names):
            
            movie_clip_featmaps, _ = test_dataset.load_movie_slidingclip(movie_name_2, 16)
            #print("number of clips: " + str(len(movie_clip_featmaps)))  # candidate clips)
            if idx2%50 == 0:
                print("video clip belongs to movie no.: %d/%d" %(idx2, all_number))
                
            # need to catenate the matrices
            if idx2 == 0:
                sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
                sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
            else:
                sentence_image_mat = np.concatenate((sentence_image_mat,np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])),axis=1)
                sentence_image_reg_mat = np.concatenate((sentence_image_reg_mat,np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps),2])),axis=1)
                
            # for each sentence
            for k in range(len(movie_clip_sentences)):
            
                # get the sent vec
                sent_vec = movie_clip_sentences[k][1]
                sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800  # why this size?
                sent_vec = torch.from_numpy(sent_vec).cuda()
    
                VP_spacy_vec = movie_clip_sentences[k][2]
                VP_spacy_vec = np.reshape(VP_spacy_vec, [1, VP_spacy_vec.shape[0]])
                VP_spacy_vec = torch.from_numpy(VP_spacy_vec).float().cuda()
                
                for t in range(len(movie_clip_featmaps)):
                    # for each sentence, slide through all the clips
                    featmap = movie_clip_featmaps[t][1] #12288 ??
                    visual_clip_name = movie_clip_featmaps[t][0]
                    softmax_ = movie_clip_featmaps[t][2]
    
                    start = float(visual_clip_name.split("_")[1])
                    end = float(visual_clip_name.split("_")[2].split("_")[0])
                    conf_score = float(visual_clip_name.split("_")[7])
    
                    featmap = np.reshape(featmap, [1, featmap.shape[0]])
                    featmap = torch.from_numpy(featmap).cuda()
    
                    softmax_ = np.reshape(softmax_, [1, softmax_.shape[0]])
                    softmax_ = torch.from_numpy(softmax_).cuda()
    
                    # network forward
                    if opt.model == 'TALL':
                        outputs = net(featmap, sent_vec)
                    elif opt.model == 'MAC':
                        outputs = net(featmap, sent_vec, softmax_, VP_spacy_vec)
    
                    outputs = outputs.squeeze(1).squeeze(1)
                    # output[0]: confidence score
                    # output[1]: offset on the starting time? 
                    # output[2]: ofset on the right side?
    
                    if opt.model == 'TALL':
                        sentence_image_mat[k, t + n_clip] = outputs[0]
                    elif opt.model == 'MAC':
                        sigmoid_output0 = 1 / float(1 + torch.exp(-outputs[0]))
                        sentence_image_mat[k, t + n_clip] = sigmoid_output0 * conf_score
    
                    # sentence_image_mat[k, t] = expit(outputs[0]) * conf_score
                    reg_end = end + outputs[2]
                    reg_start = start + outputs[1]
    
                    sentence_image_reg_mat[k, t + n_clip, 0] = reg_start
                    sentence_image_reg_mat[k, t + n_clip, 1] = reg_end
            
            n_clip += len(movie_clip_featmaps)               
            iclips = [b[0] for b in movie_clip_featmaps]
            iclips_all.append(iclips)

            
        
        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            # ********modified for corpus
            correct_num_100 = compute_IoU_recall_top_n_forreg_corpus(100, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips_all, idx1) # actual index of video that contains the sentence
            print(movie_name_1 + " IoU=" + str(IoU) + ", R@100: " + str(correct_num_100 / len(sclips)))
            
            # ************
            '''
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(
                IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
                correct_num_1 / len(sclips)))
            '''

            all_correct_num_100[k] += correct_num_100
            #all_correct_num_5[k] += correct_num_5
            #all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
        
    # for different threshold, print all testset output resullt   
    assert os.path.isdir('Charades_TALL'), 'Charades_TALL folder doesnot exist'
    test_result_output=open(os.path.join('Charades_TALL', "test_results_corpus.txt"), "w")
   
    '''
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))

        test_result_output.write("IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(
            all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")
    '''
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(all_correct_num_100[k] / all_retrievd))

        test_result_output.write("IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(
            all_correct_num_100[k] / all_retrievd))
        

    
    

if __name__ == '__main__':
    
    '''
    sample code:
        # using the best R5 model
        # for single video test: python main_charades_SL.py --model TALL --weight_path ./Charades_TALL/best_R5_IOU5_model.t7 --test_type single_video
        # for all video test: python main_charades_SL.py --model TALL --weight_path ./Charades_TALL/best_R5_IOU5_model.t7 --test_type all_video
    '''
    
    confirm_file(opt.weight_path)    
    '''
    if opt.test_type == 'single_video':
        test(opt.weight_path)
    elif opt.test_type == 'all_video':
        test_corpus(opt.weight_path)
    '''
    
    output = test_corpus_only_pairs(opt.weight_path)
    a=1