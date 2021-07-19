# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:27:23 2020

@author: tilly963
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import timedelta

## ML lib
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from sklearn.externals import joblib

from visualize.Verification import Verification 

#from model.CRNN.ConvLSTM_v2 import ConvLSTM
from data.radar_echo_NWP import load_data
from data.radar_echo_CREF_p20_out315_0824 import load_data_CREF
import time
from data.radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data

from CustomUtils_v2 import SaveSummary

import os
import shutil
import argparse
import numpy as np
import torch
#from core.data_provider import datasets_factory
from core.models.model_factory_LayerNormpy import Model
from core.utils import preprocess
#import core.trainer as trainer

#for test train.py
import os.path
import datetime
import cv2
import numpy as np
from skimage.measure import compare_ssim
from core.utils import preprocess
from visualize.visualized_pred import visualized_area_with_map
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
#parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
#parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/try_mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/try_mnist_predrnn')
parser.add_argument('--input_length', type=int, default=6)
parser.add_argument('--total_length', type=int, default=12)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--layer_norm', type=int, default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=100)#50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.01)#0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)#0.001)
parser.add_argument('--reverse_input', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
print(args)

a=args.img_width

    
def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:#50000
        eta -= args.sampling_changing_rate#1=1-0.00002
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))#(8, 9) 0~1的亂數
#    print("random_flip=",np.array(random_flip).shape)# (8, 9)
#    print("random_flip[0]=",random_flip[0])
#    random_flip[0]= [0.17844186 0.84035211 0.25251073 0.17387313 0.99843577 0.63203244
#     0.33349962 0.95222449 0.04404661]    
    true_token = (random_flip < eta)# (8, 9) 9預測的時間(T數量)
    
#    print("true_token",true_token)#(8, 9) 0 or 1
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    print("real_input_flag1 =",real_input_flag.shape)# (72, 16, 16, 16)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    #16 16 16 
    print("real_input_flag2 =",real_input_flag.shape)#(8, 9, 16, 16, 16)
#    print("real_input_flag[0] =",real_input_flag[0])
#    sys.exit()
  
    
    return eta, real_input_flag


def record_train(model, ims, real_input_flag, configs, itr, index ,num_of_batch_size, save_path, model_name):
    cost = model.train(ims, real_input_flag)#model_factory.py
    # cost = model.train(ims)#model_factory.py
    
    # if configs.reverse_input:
    #     ims_rev = np.flip(ims, axis=1).copy()
    #     print("ims_rev=",ims_rev.shape)
    #     cost += model.train(ims_rev, real_input_flag)
    #     cost = cost / 2

    if index == num_of_batch_size-1:
#        fn_path = save_path#!
        fn = model_name + '.txt'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fn = save_path + fn
        with open(fn,'a') as file_obj:
            file_obj.write( str(cost) + '\n')

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
    print('training loss: ' + str(cost))
    return cost
#  {'Banqiao': 874, 'Keelung': 1749, 'Taipei': 2625, 'New_House': 3423, 'Chiayi': 4391,
#   'Dawu': 5346, 'Hengchun': 6308, 'Success': 7185, 'Sun_Moon_Lake': 8077, 'Taitung': 8951, 'Yuxi': 9802, 
#   'Hualien': 10779, 'Beidou': 11732, 'Bao_Zhong': 12697,
#   'Chaozhou': 13665, 'News': 14629, 'Member_Hill': 15534, 'Yuli': 16467, 'Snow_Ridge': 17338, 'Shangdewen': 18299}   
def batch_sample(train_generator, places, num_of_batch_size):
    sample_p20 = []
    sample_num={}
    for place in places:
        for index in range(num_of_batch_size):
            # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
            batch_x, batch_y = train_generator.generator_sample(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), 6, 64, 64, 1)
            if batch_y.shape[0] is not 0:
                bothx_y=np.concatenate((batch_x, batch_y), axis=1)
                sample_p20.append(bothx_y)
            else :
                continue
        print("place =",place,"bothx_y=",np.array(sample_p20).shape)
        sample_num[place] = len(sample_p20)
    
    print(sample_num)
    print("sample_p20 = ",np.array(sample_p20).shape)
    sample_p20 = np.array(sample_p20).reshape(-1,12,64,64,1)
    return sample_num
        # print("bothx_y.shape=",np.array(bothx_y).shape)   





def train_sample_wrapper(model, save_path, pretrained_model, model_name, train_radar_xy_shuffle,val_radar_xy_shuffle=None):    
    from pickle import load
    train_day_path = save_path+'train_day/'
    print("===train_sample_wrapper===")
    if pretrained_model is not None:
        print('pretrained_model',str(pretrained_model))
        model.load(save_path, pretrained_model)
    if train_radar_xy_shuffle is not None:
        places=['max places'] 
        print("all train_radar_xy_shuffle shape=",train_radar_xy_shuffle.shape)
        print("all val_radar_xy_shuffle shape=",val_radar_xy_shuffle.shape)

        num_of_batch_size = len(train_radar_xy_shuffle)//args.batch_size
        val_num_of_batch_size = len(val_radar_xy_shuffle)//args.batch_size

        print("train_radar_xy_shuffle num_of_batch_size=",num_of_batch_size)
        print("val num_of_batch_size=",val_num_of_batch_size)
    
    else:
        train_generator = radar.generator('train', batch_size=args.batch_size,save_path=train_day_path)
        num_of_batch_size = train_generator.step_per_epoch#!-1
    places=['Sun_Moon_Lake'] 
    place_len=len(places)
    # print("places=",places)
    # print("range(train_generator.step_per_epoch)=",range(train_generator.step_per_epoch))
    eta = args.sampling_start_value
    patience = 5
    min_loss = 100
    # patience = 2
    trigger_times = 0
    sample_p20 =[]
    print("place_len = ",place_len)
    for itr in range(1,2500):
        # model.train()
        # print("model.training() = ",model.training())
        print("===========itr=",itr,"===========")
        # print("all train_radar_xy_shuffle shape=",train_radar_xy_shuffle.shape)
        sum_cost = 0
        avg_cost =0
        sum_xy_len=0
        sum_xy=np.zeros(0)
        smaple_p20_number = 0
        print("num_of_batch_size = ",num_of_batch_size)
        for place in places:
            smaple_number = 0
            cost_p1 = 0
            avg_p1_cost = 0
            print("place",place)#,"sample_num[place]",sample_num[place] )
            # num_of_batch_size = sample_num[place]
            for index in range(num_of_batch_size):
                batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
                # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
                # batch_x, batch_y = train_generator.generator_sample(index, place)
                # train_xy = train_radar_xy_shuffle[index*args.batch_size:(index+1)*args.batch_size,:,:,:,:]#!
                print("num_of_batch_size * args.batch_size=",num_of_batch_size*args.batch_size,"index*args.batch_size=",index*args.batch_size,"to",(index+1)*args.batch_size)

                # batch_x,batch_y = np.split(train_xy, 2, axis=1)

                batch_x = batch_x.astype(np.float16)  
                batch_y = batch_y.astype(np.float16)
                
                # batch_x = np.array(batch_x).reshape(6,64*64)
                # scaler = StandardScaler()
                # batch_x=scaler.fit_transform(batch_x)
                # transformer = Normalizer().fit(batch_x)  # fit does nothing.
                # batch_x = transformer.transform(batch_x)
                # batch_x=np.array(batch_x).reshape(-1,6,64,64,1)


                scaler_path = save_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
                if not os.path.isdir(scaler_path):
                    os.makedirs(scaler_path)
                # scaler = load(open('min_max_scaler_8_240110.pkl', 'rb')) 
                # normalizer_scaler_8_240110
                '''
                scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb')) 
                # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   
                #    period=model_parameter['period'],
                #     predict_period=model_parameter['predict_period'],
                batch_x = np.array(batch_x).reshape(-1,64*64)
                batch_x = scaler.transform(batch_x)
                '''
                batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)
                batch_y=np.array(batch_y).reshape(-1,model_parameter['predict_period'],512,512,1)

                bothx_y=np.concatenate((batch_x, batch_y), axis=1)
                ims = bothx_y

                ims = preprocess.reshape_patch(ims, args.patch_size)
#                print("np.array(ims==patch_tensor).shape=",np.array(ims).shape)
                #np.array(ims==patch_tensor).shape= (8, 20, 16, 16, 16)
#                [batch_size, seq_length,img_height//patch_size, patch_size,img_width//patch_size, patch_size,num_channels]
                eta, real_input_flag = schedule_sampling(eta, itr)#!

#                trainer.train(model, ims, real_input_flag, args, itr, index ,num_of_batch_size, save_path, model_name)
                cost = record_train(model, ims, real_input_flag, args, itr, index ,num_of_batch_size, save_path, model_name)
                # cost = record_train(model, ims, args, itr, index ,num_of_batch_size, save_path, model_name)
                
                sum_xy=np.zeros(0)
                cost_p1 = cost_p1 + cost
    
            avg_p1_cost = cost_p1/num_of_batch_size#(smaple_number//32)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            fn = model_name + 'p20.txt'
            fn = save_path + fn
            with open(fn,'a') as file_obj:
                # file_obj.write('itr:' + str(itr) + 'place = '+str(place) +' training loss: ' + str(avg_p1_cost) + '\n')#應該不用除批次量
                file_obj.write(str(avg_p1_cost) + '\n')#應該不用除批次量
                
                # file_obj.write('num_of_batch_size*1:' + str(num_of_batch_size*1)+'\n') #!
                
                # file_obj.write('sample_num[place]:' + str(smaple_number)+'\n') 
                
            sum_cost = sum_cost + avg_p1_cost
            smaple_p20_number = smaple_p20_number + smaple_number

        avg_cost = sum_cost/place_len
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        fn = model_name + 'avg_p20.txt'
        fn = save_path + fn
        with open(fn,'a') as file_obj:
            file_obj.write('itr:' + str(itr) + 'place20,  training loss: ' + str(avg_cost) + '\n')#應該不用除批次量
            file_obj.write('smaple_p20_number:' + str(smaple_p20_number) + '\n')#應該不用除批次量

        model_pkl = model_name+'_itr{}.pkl'.format(itr)
        # if itr%100 == 0:
            # model.save(model_pkl,save_path) # val(model, save_path, model_pkl, itr)
        load_model=False
        test_cost, ssim = val(model, save_path, model_pkl, itr,val_radar_xy_shuffle=None)
        # test_cost = test_wrapper(model, save_path, model_pkl, itr,load_model=load_model)
        # if test_cost <= 0.001 or itr == 30000 or itr%100==0 :
        model_pkl = model_name+'_itr{}_test_cost{}_ssim{}.pkl'.format(itr, test_cost,ssim)
        model.save(model_pkl,save_path) # val(model, save_path, model_pkl, itr)
        if test_cost > min_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                model_pkl = model_name+'_itr{}_earlystopping{}_test_cost{}_min_loss{}.pkl'.format(itr, patience, test_cost, min_loss)
                model.save(model_pkl,save_path)
                # return model
        else:
            trigger_times = 0
            print('trigger times: ',trigger_times)

            min_loss = test_cost


def train_wrapper(model,save_path,model_name):    
    from pickle import load
    train_day_path = save_path+'train_day/'
    train_generator = radar.generator('train', batch_size=1,save_path=train_day_path)
    print("places=",places)
    print("range(train_generator.step_per_epoch)=",range(train_generator.step_per_epoch))
    eta = args.sampling_start_value
    num_of_batch_size = train_generator.step_per_epoch#!-1
    # sample_num = batch_sample(train_generator,places, num_of_batch_size)
    # print("sample_p20 ",np.array(sample_p20).shape)
    # print("sample_p20/32=",len(sample_p20)/32)
    # sys.exit()
    patience = 100
    min_loss = 100
    # patience = 2
    trigger_times = 0
    sample_p20 =[]
    place_len=len(places)
    print("place_len = ",place_len)
 
    for itr in range(1,40001):
        # model.train()
        # print("model.training() = ",model.training())
        print("===========itr=",itr,"===========")
        sum_cost = 0
        avg_cost =0
        sum_xy_len=0
        sum_xy=np.zeros(0)
        smaple_p20_number = 0
        print("num_of_batch_size = ",num_of_batch_size)
        for place in places:
            smaple_number = 0
            cost_p1 = 0
            avg_p1_cost = 0
            print("place",place)#,"sample_num[place]",sample_num[place] )
            # num_of_batch_size = sample_num[place]
            for index in range(num_of_batch_size):
                # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
                # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
                batch_x, batch_y = train_generator.generator_sample(index, place)

                batch_x = batch_x.astype(np.float16)  
                batch_y = batch_y.astype(np.float16)
                
                # batch_x = np.array(batch_x).reshape(6,64*64)
                # scaler = StandardScaler()
                # batch_x=scaler.fit_transform(batch_x)
                # transformer = Normalizer().fit(batch_x)  # fit does nothing.
                # batch_x = transformer.transform(batch_x)
                # batch_x=np.array(batch_x).reshape(-1,6,64,64,1)
                batch_y = batch_y.reshape(len(batch_y), 6, 512, 512, 1)

                if batch_y.shape[0] is 0:
                    print("batch_x is zero(<5).shape=",batch_x.shape)#(1, 6, 64, 64, 1)
                    continue
                else:
                    print("batch_x.shape=",batch_x.shape)#(1, 6, 64, 64, 1)
                    print("batch_y.shape=",batch_y.shape)#(1, 6, 64, 64, 1)
                    bothx_y_temp = np.concatenate((batch_x, batch_y), axis=1)
                    smaple_number+=1
                if sum_xy.shape[0] is 0:#第一次
                    print("creat sum_xy before =",sum_xy.shape)
                    sum_xy = bothx_y_temp
                    print("creat sum_xy after =",sum_xy.shape)
                    continue
                if sum_xy.shape[0] < 32 and sum_xy is not False:#第二次到31次
                    # print("add sum_xy before =",sum_xy.shape)(31, 12, 64, 64, 1)
                    sum_xy = np.concatenate((bothx_y_temp, sum_xy), axis=0)
                    # print("add sum_xy after =",sum_xy.shape)(32, 12, 64, 64, 1)
                    continue

                print("count 32 sum_xy.shape=",np.array(sum_xy).shape)
                # ims = sum_xy
                batch_x,batch_y = np.split(sum_xy, 2, axis=1)

                scaler_path = save_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
                if not os.path.isdir(scaler_path):
                    os.makedirs(scaler_path)
                # scaler = load(open('min_max_scaler_8_240110.pkl', 'rb')) 
                # normalizer_scaler_8_240110
                scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb')) 
                # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   

                batch_x = np.array(batch_x).reshape(-1,64*64)
                batch_x = scaler.transform(batch_x)
                batch_x=np.array(batch_x).reshape(-1,6,64,64,1)

                bothx_y=np.concatenate((batch_x, batch_y), axis=1)
                ims = bothx_y

                ims = preprocess.reshape_patch(ims, args.patch_size)
            #    print("np.array(ims==patch_tensor).shape=",np.array(ims).shape)
            #     np.array(ims==patch_tensor).shape= (8, 20, 16, 16, 16)
            #    [batch_size, seq_length,img_height//patch_size, patch_size,img_width//patch_size, patch_size,num_channels]
                eta, real_input_flag = schedule_sampling(eta, itr)

            #    trainer.train(model, ims, real_input_flag, args, itr, index ,num_of_batch_size, save_path, model_name)
                cost = record_train(model, ims, real_input_flag, args, itr, index ,num_of_batch_size, save_path, model_name)
                sum_xy=np.zeros(0)
                cost_p1 = cost_p1 + cost
    
            avg_p1_cost = cost_p1/num_of_batch_size#(smaple_number//32)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            fn = model_name + 'p20.txt'
            fn = save_path + fn
            with open(fn,'a') as file_obj:
                file_obj.write('itr:' + str(itr) + 'place = '+str(place) +' training loss: ' + str(avg_p1_cost) + '\n')#應該不用除批次量
                file_obj.write('num_of_batch_size*1:' + str(num_of_batch_size*1)+'\n') #!
                
                file_obj.write('sample_num[place]:' + str(smaple_number)+'\n') 
                
            sum_cost = sum_cost + avg_p1_cost
            smaple_p20_number = smaple_p20_number + smaple_number

        avg_cost = sum_cost/place_len
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        fn = model_name + 'avg_p20.txt'
        fn = save_path + fn
        with open(fn,'a') as file_obj:
            file_obj.write('itr:' + str(itr) + 'place20,  training loss: ' + str(avg_cost) + '\n')#應該不用除批次量
            file_obj.write('smaple_p20_number:' + str(smaple_p20_number) + '\n')#應該不用除批次量

        model_pkl = model_name+'_itr{}.pkl'.format(itr)
        # if itr%100 == 0:
            # model.save(model_pkl,save_path) # val(model, save_path, model_pkl, itr)
        load_model=False
        test_cost = val(model, save_path, model_pkl, itr)
        # test_cost = test_wrapper(model, save_path, model_pkl, itr,load_model=load_model)
        # if test_cost <= 0.001 or itr == 30000 or itr%100==0 :
        model_pkl = model_name+'_itr{}_test_cost{}.pkl'.format(itr, test_cost)
        model.save(model_pkl,save_path) # val(model, save_path, model_pkl, itr)
        if test_cost > min_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                model_pkl = model_name+'_itr{}_earlystopping{}_test_cost{}_min_loss{}.pkl'.format(itr, patience, test_cost, min_loss)
                model.save(model_pkl,save_path)
                # return model
        else:
            trigger_times = 0
            print('trigger times: ',trigger_times)

        min_loss = test_cost
            
def csi_picture(img_out, test_ims, save_path,data_name='csi'):
        test_x_6 = test_ims
        img_out = img_out#t9~t18
        if not os.path.isdir(save_path):
            os.makedirs(save_path)       
        
        Color = ['#00FFFF', '#4169E1', '#0000CD', '#ADFF2F', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#9932CC']

        ## CSI comput
        csi = []
        img_out_315 = img_out[:,:,97:412,97:412,:]
        print("1 floa32 img_out_315.dtype=",img_out_315.dtype)  
        # img_out_315[img_out_315 <= 1] = 0

        test_x_6_315 = test_x_6[:,:,97:412,97:412,:]
        print("2 test_x_6_315=",np.array(test_x_6_315).shape)
        
        img_out_315 = img_out_315.astype(np.float16)
        test_x_6_315 = test_x_6_315.astype(np.float16)
        print("2 floa32 img_out_315.dtype=",img_out_315.dtype)  

        print("2 test_x_6_315=",np.array(test_x_6_315).shape)
        img_out_315_0 = np.array(img_out_315[0,0,:,:,:]).reshape(315,315)
        test_x_6_315_0 =np.array(test_x_6_315[0,0,:,:,:]).reshape(315,315)
        print("img_out_315_0=",np.array(img_out_315_0).shape)

        visualized_area_with_map(img_out_315_0, 'Sun_Moon_Lake', shape_size=[315,315], title='pred_to_010', savepath=save_path)
        visualized_area_with_map(test_x_6_315_0, 'Sun_Moon_Lake', shape_size=[315,315], title='test_to_010', savepath=save_path)



        img_out_315_1 = np.array(img_out_315[0,1,:,:,:]).reshape(315,315)
        test_x_6_315_1 = np.array(test_x_6_315[0,1,:,:,:]).reshape(315,315)
        print("img_out_315_1=",np.array(img_out_315_1).shape)

        visualized_area_with_map(img_out_315_1, 'Sun_Moon_Lake', shape_size=[315,315], title='pred_to_020', savepath=save_path)
        visualized_area_with_map(test_x_6_315_1, 'Sun_Moon_Lake', shape_size=[315,315], title='test_to_020', savepath=save_path)

        for period in range(model_parameter['predict_period']):
            print("period=",period)
        #    print('pred_y[:, period] = ', pred_y[:, period])
        #    print('test_y[:, period] = ', test_y[:, period])
            csi_eva = Verification(pred=img_out_315[:, period].reshape(-1, 1), target=test_x_6_315[:, period].reshape(-1, 1), threshold=60, datetime='')
            print("csi_eva.csi.shape=",np.array(csi_eva.csi).shape)# (60, 99225)
            print("csi_eva.csi")
            print(csi_eva.csi)
            csi.append(np.nanmean(csi_eva.csi, axis=1))
            print("csi")

            print(csi)
            print("csi_eva.csi[0,:]")
            print(csi_eva.csi[0,:])
            print("mean",np.mean(csi_eva.csi[0,np.isfinite(csi_eva.csi[0,:])]))
            print("np.array(csi).shape=",np.array(csi).shape)#(1, 60)
            # sys.exit()
        
        csi = np.array(csi)
        np.savetxt(save_path+'{}.csv'.format(data_name), csi, delimiter = ',')
        # np.savetxt(save_path+'T202005270000csi.csv', csi.reshape(6,60), delimiter = ' ')

        ## Draw thesholds CSI
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)

        all_csi = []
        for period in range(model_parameter['predict_period']):
            plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), 'o--', label='{} min'.format((period+1)*10))

        plt.legend(loc='upper right')

        fig.savefig(fname=save_path+'Thresholds_CSI.png', format='png')
        plt.clf()


        ## Draw thesholds AVG CSI
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)

        all_csi = []
        plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), 'o--', label='AVG CSI')
        
        plt.legend(loc='upper right')

        fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
        plt.clf()

        #csie = time.clock()
        #
        #alle = time.clock()
        #
        #print("load NWP time = ", loadNe - loadNs)
        #print("load CREF time = ", loadCe - loadCs)
        #print("All time = ", alle - alls)
        ## Draw peiod ALL CSI 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, model_parameter['predict_period']+1)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Time/10min')
        plt.ylabel('CSI')
        my_x_ticks = np.arange(0, model_parameter['predict_period']+1, 1)
        plt.xticks(my_x_ticks)
        plt.title('Threshold 5-55 dBZ')
        plt.grid(True)
        i = 0
        for threshold in range(5, 56, 5):
            plt.plot(np.arange(len(csi)+1), [np.nan] + list(csi[:, threshold-1]), 'o--', label='{} dBZ'.format(threshold), color=Color[i])
            i = i + 1
        #plt.legend(loc='lower right')

        plt.clf()

        fig.savefig(fname=save_path+'Period_CSI_ALL2.png', format='png')

        rmse_315=np.sqrt(((img_out_315 - test_x_6_315) ** 2).mean())

        rmse=np.sqrt(((img_out - test_x_6) ** 2).mean())

        fn = save_path + '{}_rmse.txt'.format(data_name)
        with open(fn,'a') as file_obj:
            file_obj.write('rmse=' + str(rmse)+'\n')

            file_obj.write('rmse_315=' + str(rmse_315)+'\n')
       

            # file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
            # file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            # file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*1)  + '\n' )
            # file_obj.write("place" + str(place)  + '\n' )
            
            # for i in range(args.total_length - args.input_length):
            #     print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * 1))#
            #     print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * 1))/(512*512)))
            #     # file_obj.write("avg 512x512 mse seq[" + str(i) + '],test loss: ' + str((img_mse[i] /32)/(512*512)) + '\n')  
            #     file_obj.write("avg 512x512_2 mse seq[" + str(i) + '],test loss: ' + str(((img_mse[i] /1)/num_of_batch_size)/(512*512)) + '\n')  
            #     file_obj.write("avg 512x512_2 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]/batch_id * 1) + '\n')  

            # # file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
            # file_obj.write("test loss:" + str(avg_batch_cost) + '\n')     
            # file_obj.write("test mse:" + str(avg_mse_p1) + '\n')  
            # file_obj.write("test avg_ssim:" + str(avg_ssim) + '\n')  
def test_wrapper(model, save_path, model_pkl, itr,load_model=False ):
    print("===========test_wrapper===========")
    # model.eval()
    from pickle import load

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if load_model:
        model.load(save_path, model_name)
    place_len = len(places)
    main_path = save_path
    # save_path = save_path + 'test_wrapper_itr{}_20205M9D_2018313/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_201803130010to03132359/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_201808240010to08242359_v3/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_202005270010to05272359_v3/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_202005270010to20_v3/'.format(itr)
    save_path = save_path + 'test_wrapper_itr{}_201808240010to20_v4/'.format(itr)
    # 
    # save_path = save_path + 'test_wrapper_itr{}_201808240010_v3/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_201803130010_v3/'.format(itr)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # model.load_state_dict(torch.load('params.pkl')) 
    # places=['Bao_Zhong']
    # date_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
    # test_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
    # date_date=[['2018-03-13 00:10', '2018-03-13 00:19']]
    # test_date=[['2018-03-13 00:10', '2018-03-13 00:19']]

    # date_date=[['2018-08-24 00:10', '2018-08-24 23:59']]
    # test_date=[['2018-08-24 00:10', '2018-08-24 23:59']]
    date_date=[['2018-08-24 00:10', '2018-08-24 00:29']]
    test_date=[['2018-08-24 00:10', '2018-08-24 00:29']]

    # date_date=[['2020-05-27 00:10', '2020-05-27 23:59']]
    # test_date=[['2020-05-27 00:10', '2020-05-27 23:59']]
    # date_date=[['2020-05-27 00:10', '2020-05-27 00:29']]
    # test_date=[['2020-05-27 00:10', '2020-05-27 00:29']]

    # places=['Banqiao','Keelung','Taipei','New_House','Chiayi',
    #     'Dawu','Hengchun','Success','Sun_Moon_Lake','Taitung',#,
    #     'Yuxi','Hualien','Beidou','Bao_Zhong','Chaozhou',
    #     'News','Member_Hill','Yuli','Snow_Ridge','Shangdewen']
    #        5/16、19、21、22、23、26、27、28、29
    # # places=['Banqiao','Keelung']
    '''
    date_date=[['2018-03-13 00:00','2018-08-13 23:59'],
        ['2020-05-16 00:00','2020-05-16 23:59'],
        ['2020-05-19 00:00','2020-05-19 23:59'],
        ['2020-05-21 00:00','2020-05-21 23:59'],
        ['2020-05-22 00:00','2020-05-22 23:59'],
        ['2020-05-23 00:00','2020-05-23 23:59'],
        ['2020-05-26 00:00','2020-05-26 23:59'],
        ['2020-05-27 00:00','2020-05-27 23:59'],
        ['2020-05-28 00:00','2020-05-28 23:59'],
        ['2020-05-29 00:00','2020-05-29 23:59']]
    val_data=[['2018-03-13 00:00','2018-08-13 23:59'],
    ['2020-05-16 00:00','2020-05-16 23:59'],
            ['2020-05-19 00:00','2020-05-19 23:59'],
            ['2020-05-21 00:00','2020-05-21 23:59'],
            ['2020-05-22 00:00','2020-05-22 23:59'],
            ['2020-05-23 00:00','2020-05-23 23:59'],
            ['2020-05-26 00:00','2020-05-26 23:59'],
            ['2020-05-27 00:00','2020-05-27 23:59'],
            ['2020-05-28 00:00','2020-05-28 23:59'],
            ['2020-05-29 00:00','2020-05-29 23:59']]
    '''
    radar_echo_storage_path= 'D:/yu_ting/try/NWP/'#'NWP/'
    load_radar_echo_df_path=main_path+'2018_7mto8m_Sun_Moon_Lake_512x512_T12toT6.pkl'
    radar_test = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                    load_radar_echo_df_path=load_radar_echo_df_path,#'data/RadarEcho_Bao_Zhong_2018_08240010_T6toT6_inoutputshape64_random.pkl',#None,#load_radar_echo_df_path,
                    input_shape=[512,512],#[512,512],#model_parameter['input_shape'],
                    output_shape=[512,512],#model_parameter['output_shape'],
                    period=model_parameter['period'],
                    predict_period=model_parameter['predict_period'],
                    places=places,
                    random=False,
                    date_range=date_date,
                    test_date=test_date)
    if not load_radar_echo_df_path:
        radar_test.exportRadarEchoFileList()
    #     radar_tw.saveRadarEchoDataFrame()
        radar_test.saveRadarEchoDataFrame(path=save_path ,load_name_pkl='2020_9day_20180313_Sun_Moon_Lake_512x512')   

    
    test_day_path = save_path+'test_day/'

    test_generator = radar_test.generator('test', batch_size=1)#)args.batch_size,save_path = test_day_path )
   
    batch_id = 0
    img_mse, ssim = [], []

    real_input_flag = np.zeros(
        (1,#args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    p20_cost =0
    avg_p20_cost=0

    p20_mse =0
    for place in places: 
        print("places=",places)
        print("range(test_generator.step_per_epoch)=",range(test_generator.step_per_epoch))
        num_of_batch_size = test_generator.step_per_epoch#!-1
        batch_id = 0
        batch_cost = 0
        avg_batch_cost = 0
        
        sum_mse_index = 0
        img_mse, ssim = [], []
        avg_ssim=0
        for i in range(args.total_length - args.input_length):
            img_mse.append(0)
            ssim.append(0)
        for index in range(num_of_batch_size):
            batch_id = batch_id + 1
            batch_x, batch_y = test_generator.generator_getClassifiedItems_3(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)
            scaler_path = main_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            if not os.path.isdir(scaler_path):
                os.makedirs(scaler_path)
            # scaler = load(open('min_max_scaler_8_240210.pkl', 'rb'))       
            # srandard
            # scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb'))       
            # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   
            # batch_x = np.array(batch_x).reshape(6,512*512)
            # batch_x = scaler.transform(batch_x)
            batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)


            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            val_ims = np.stack(bothx_y,axis=0)   
            print("np.array(val_ims).shape=",np.array(val_ims).shape)

            val_dat = preprocess.reshape_patch(val_ims, args.patch_size)
            print("test_dat  preprocess.reshape_patch=",np.array(val_dat).shape)
            img_gen = model.test(val_dat, real_input_flag)
            print("--預測-")
            print("img_gen model.test=",np.array(img_gen).shape)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)
            output_length = args.total_length - args.input_length
            # img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]
            # img_out_all.append(img_out)
            test_x_6 = val_ims[:, model_parameter['period']:, :, :, :]
            # test_x_6_all.append(test_x_6)
            if batch_id ==1:
                test_ims_all = test_x_6
                img_out_all = img_out
            else:
                test_ims_all = np.concatenate((test_ims_all , test_x_6) ,axis = 0)#old nee
                img_out_all = np.concatenate((img_out_all , img_out) ,axis = 0)#old nee
            
            print("test_ims_all",np.array(test_ims_all).shape)
            print("img_out_all",np.array(img_out_all).shape)

            mse = np.square(test_x_6 - img_out).sum()
            mse_picture_avg = ((mse/1)/model_parameter['predict_period'])/(512*512)
            sum_mse_index = sum_mse_index + mse_picture_avg
            # MSE per frame
            save_path_single_location = save_path+'512x512_test_9d/'
            seq_p1_cost =0
            sum_mse =0
            avg_seq_p1_cost= 0
            for i in range(output_length):
                x = val_ims[:, i + args.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # gx = np.maximum(gx, 0)
                # gx = np.minimum(gx, 60)
                print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (8, 512, 512, 1) gx= (8, 512, 512, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                # avg_mse += mse
                sum_mse +=mse
                account_mse = sum_mse/1

                # account_mse = mse/1
                seq_p1_cost = seq_p1_cost + account_mse 
                print("batch_id=",batch_id,"i + args.input_length=",i + args.input_length,"account_mse=",account_mse)
                vis_x = x.copy()
                # vis_x[vis_x > 1.] = 1.
                # vis_x[vis_x < 0.] = 0.
                vis_gx = gx.copy()
                vis_x = vis_x/65
                vis_gx = vis_gx/65
                # vis_gx[vis_gx > 1.] = 1.
                # vis_gx[vis_gx < 0.] = 0.

                real_frm = np.uint8(vis_x * 255).reshape(512,512)
                pred_frm = np.uint8(vis_gx * 255).reshape(512,512)

                # for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm, real_frm,data_range=255, full=True, multichannel=True)
                ssim[i] += score     
                avg_ssim+=score
            avg_seq_p1_cost = ((sum_mse/1)/model_parameter['predict_period'])/(512*512)

            batch_cost = batch_cost + avg_seq_p1_cost

        avg_batch_cost =batch_cost/num_of_batch_size
        avg_mse_p1 = sum_mse_index/num_of_batch_size
        # print('mse per seq: ' + str(avg_mse))    
        print("ssim",np.array(ssim).shape)
        avg_ssim = (avg_ssim/model_parameter['predict_period'])/num_of_batch_size
        save_path_index = save_path_single_location + '512x512_PD/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        if not os.path.isdir(save_path_index):
            os.makedirs(save_path_index)
        fn = save_path_index + 'test_mse_div20_.txt'
        with open(fn,'a') as file_obj:
            file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
            file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*1)  + '\n' )
            file_obj.write("place" + str(place)  + '\n' )
            
            for i in range(args.total_length - args.input_length):
                print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * 1))#
                print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * 1))/(512*512)))
                # file_obj.write("avg 512x512 mse seq[" + str(i) + '],test loss: ' + str((img_mse[i] /32)/(512*512)) + '\n')  
                file_obj.write("avg 512x512_2 mse seq[" + str(i) + '],test loss: ' + str(((img_mse[i] /1)/num_of_batch_size)/(512*512)) + '\n')  
                file_obj.write("avg 512x512_2 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]/batch_id * 1) + '\n')  

            # file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
            file_obj.write("test loss:" + str(avg_batch_cost) + '\n')     
            file_obj.write("test mse:" + str(avg_mse_p1) + '\n')  
            file_obj.write("test avg_ssim:" + str(avg_ssim) + '\n')  


        p20_mse = p20_mse+ avg_mse_p1
        p20_cost = p20_cost + avg_batch_cost
    p20_mse = p20_mse/place_len
    avg_p20_cost = p20_cost/place_len
    # avg_ssim = avg_ssim/place_len#!
    fn = save_path_single_location + 'test_mse_avg20_.txt'
    with open(fn,'a') as file_obj:
        file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
        file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*1)  + '\n' )
        file_obj.write("place 20"   + '\n' )
        file_obj.write("test avg_ssim:" + str(avg_ssim) + '\n')  

        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(512*512))/4)
        #   file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write("test loss:" + str(avg_p20_cost) + '\n') 

        file_obj.write("p20_mse mse:" + str(p20_mse) + '\n')  



    fn = save_path_single_location + 'test_mse_itr.txt'
    with open(fn,'a') as file_obj:
        # file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
        # file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        # file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
        # file_obj.write("place 20"   + '\n' )
        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
        #   file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write(str(p20_mse) + '\n') 

        # file_obj.write("p20_mse mse:" + str(p20_mse) + '\n')  
  
    csi_picture(img_out = img_out_all,test_ims= test_ims_all,save_path = save_path+'csi/')
    return p20_mse

def test_show(model, save_path, model_name,itr):
    print("===========test_show===========")

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.load(save_path, model_name)
    main_path = save_path
    # save_path = save_path + 'test_show_itr_{}_201808240010_Sun_Moon_Lake_dbz1_315X315/'.format(itr)
    # save_path = save_path + 'test_show_itr_{}_202005190010_Sun_Moon_Lake_dbz1/'.format(itr)

    # save_path = save_path + 'test_show_itr_{}_202005270010to5272359_Sun_Moon_Lake_dbz1_315X315/'.format(itr)
    # data_name='201905170010'
    # data_name='202005270010'
    # data_name='202005160800to6'
    # data_name='202005280000to6'
    # data_name='202005270000to6'
    
    # data_name='201808240000to6'
    # data_name='202008260400to6'
    # data_name='202005220400to6'
    data_name='202005290600to6'

    

    save_path = save_path + 'test_show_itr_{}_{}_Sun_Moon_Lake_dbz1_csi_testcsi/'.format(itr,data_name)
    from pickle import load
    # places=['Bao_Zhong']
    # date_date=[['2018-08-23 22:10', '2018-08-24 03:59']]
    # test_date=[['2018-08-24 00:10', '2018-08-24 01:00']]
    # test_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
    # date_date=[['2020-05-20 22:10', '2020-05-21 02:59']]
    # test_date=[['2020-05-21 00:10', '2020-05-21 00:11']]
    # date_date=[['2018-08-23 00:00', '2018-08-23 23:59']]
    # test_date=[['2018-08-23 01:10', '2018-08-23 01:11']]
    places=['Sun_Moon_Lake']
    place_len=len(places)
    # date_date=[['2018-06-15 12:00', '2018-06-15 12:01']]
    # test_date=[['2018-06-15 12:00', '2018-06-15 12:01']]
    # date_date=[['2018-08-24 00:00', '2018-08-24 23:59']]

    # test_date=[['2018-08-24 00:10', '2018-08-24 00:11']]
    # date_date=[['2018-08-24 00:10', '2018-08-24 00:11']]
    
    # test_date=[['2020-08-26 04:10', '2020-08-26 04:11']]
    # date_date=[['2020-08-26 04:10', '2020-08-26 04:11']]

    # test_date=[['2020-05-22 04:10', '2020-05-22 04:11']]
    # date_date=[['2020-05-22 04:10', '2020-05-22 04:11']]

    test_date=[['2020-05-29 06:10', '2020-05-29 06:11']]
    date_date=[['2020-05-29 06:10', '2020-05-29 06:11']]

    # test_date=[['2018-08-24 03:10', '2018-08-24 03:11']]
    # date_date=[['2018-05-09 00:00', '2018-05-09 00:01']]
    # test_date=[['2018-05-09 00:00', '2018-05-09 00:01']]


    # date_date=[['2020-05-27 00:10', '2020-05-27 00:19']]
    # test_date=[['2020-05-27 00:10', '2020-05-27 00:19']]


    # date_date=[['2020-08-26 00:10', '2020-08-26 00:19']]
    # test_date=[['2020-08-26 00:10', '2020-08-26 00:19']]

    # date_date=[['2019-05-17 00:10', '2019-05-17 00:19']]
    # test_date=[['2019-05-17 00:10', '2019-05-17 00:19']]


    # date_date=[['2020-05-19 00:10', '2020-05-19 00:19']]
    # test_date=[['2020-05-19 00:10', '2020-05-19 00:19']]

    # date_date=[['2020-05-16 08:10', '2020-05-16 08:11']]
    # test_date=[['2020-05-16 08:10', '2020-05-16 08:11']]


    # date_date=[['2020-05-28 00:10', '2020-05-28 00:11']]
    # test_date=[['2020-05-28 00:10', '2020-05-28 00:11']]

    # date_date=[['2018-08-23 00:10', '2018-08-23 00:10']]
    # test_date=[['2018-08-23 00:10', '2018-08-23 00:10']]
    load_radar_echo_df_path=None#"T12toT6_['Sun_Moon_Lake']_512x512_2018_3m_pretrain/201808240000to6_512x512.pkl"#None#"T12toT6_['Sun_Moon_Lake']_512x512_2018_3m/2018_7mto8m_Sun_Moon_Lake_512x512_T12toT6.pkl"#None#"T18toT6_['Sun_Moon_Lake']_512x512_v2/save_2018_801to830/2018_m8_Sun_Moon_Lake_512x512_T18toT6.pkl"#None#"T10toT10_['Sun_Moon_Lake']_512x512/save_2018_801to830/2018_1m_Sun_Moon_Lake_512x512.pkl"# None#'T10toT10_sample_1025/2018_3m_p20_inputsize64_T10toT10random_noval.pkl'#'T10toT10_sample/save_2017_2018_14m_p2_sample_/2017_2018_14m_p2_inputsize64_T10toT10random.pkl'#None#'T10toT10_sample1000/save_2018_7m_p2_sample/2018_7m_p2_inputsize64_T10toT10random.pkl'#None#'samping_test/save_2019_2018_2019_3m_p20_sample6/2018_2018_2019_3m_p20_inputsize64_random.pkl'#None#'sample_p20/save_2018_3m_p4_sample/2018_3m_p4_random.pkl'#None#'sample/save_2018_3m_sample/2018_20day.pkl'#None#'data/RadarEcho_p20_2018_2019_6mto8m_T6toT6_inoutputshape64_random.pkl'
    
    radar_echo_storage_path= 'D:/yu_ting/try/NWP/'#'NWP/'
    # load_radar_echo_df_path='data/RadarEcho_p20_2018_2019_6mto8m_T6toT6_inoutputshape64_random.pkl'
    # load_radar_echo_df_path="T18toT6_['Sun_Moon_Lake']_512x512_v2/save_2018_801to830/2018_m8_Sun_Moon_Lake_512x512_T18toT6.pkl"#None#'samping_test/save_2019_2018_2019_3m_p20_sample6/2018_2018_2019_3m_p20_inputsize64_random.pkl'#None#'sample_p20/save_2018_3m_p4_sample/2018_3m_p4_random.pkl'#None#'sample/save_2018_3m_sample/2018_20day.pkl'#None#'data/RadarEcho_p20_2018_2019_6mto8m_T6toT6_inoutputshape64_random.pkl'
    # D:\yu_ting\predrnn\predrnn_gogo\T10toT10_['Sun_Moon_Lake']_512x512\save_2018_801to830
    radar_1p = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                    load_radar_echo_df_path=load_radar_echo_df_path,#'data/RadarEcho_Bao_Zhong_2018_08240010_T6toT6_inoutputshape64_random.pkl',#None,#load_radar_echo_df_path,
                    input_shape=[512,512],#[512,512],#model_parameter['input_shape'],
                    output_shape=[512,512],#model_parameter['output_shape'],
                    period=model_parameter['period'],
                    predict_period=model_parameter['predict_period'],
                    places=places,
                    random=False,
                    date_range=date_date,
                    test_date=test_date,
                    save_np_radar=save_path )
    if not load_radar_echo_df_path:
        radar_1p.exportRadarEchoFileList()
    #     radar_tw.saveRadarEchoDataFrame()
        radar_1p.saveRadarEchoDataFrame(path=save_path ,load_name_pkl='{}_512x512'.format(data_name))   

    


    test_show_day_path = save_path+'teat_show_day/'
    test_generator = radar_1p.generator('test', batch_size=1,save_path = test_show_day_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []
    test_x_6 =[]
    img_out=[]
    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (1,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    num_of_batch_size = test_generator.step_per_epoch
    avg_ssim=0
    test_ims_all=[]
    img_out_all=[]
    sum_p20_mse_picture_avg=0
    for place in places: 
        print("places=",places)
        print("range(test_generator.step_per_epoch)=",range(test_generator.step_per_epoch))
        sum_p1_mse_picture_avg=0
        sum_mse_index =0
        batch_cost=0
        batch_id=0
        for index in range(num_of_batch_size):
            batch_id = batch_id + 1
            batch_x, batch_y = test_generator.generator_getClassifiedItems_3(index, place)
            # batch_x, batch_y = test_generator.generator_sample(index, place)  

            # batch_x = np.array(batch_x).reshape(6,64*64)

            # batch_x = scaler.transform(batch_x)
            '''
            transformer = Normalizer().fit(batch_x)
            batch_x = transformer.transform(batch_x)

            batch_x=np.array(batch_x).reshape(-1,6,64,64,1)
            '''
            
            # scaler = load(open('min_max_scaler_8_240210.pkl', 'rb'))       
            # srandard
            scaler_path = main_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            '''
            if not os.path.isdir(scaler_path):
                os.makedirs(scaler_path)
            scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb'))       
            # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   
            '''
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            np.savetxt(save_path+'radar_8_240210_testx.txt', batch_x.reshape(-1), delimiter=' ')
            np.savetxt(save_path+'radar_8_240210_testy.txt', batch_y.reshape(-1), delimiter=' ')
            '''
            batch_x = np.array(batch_x).reshape(model_parameter['period'],64*64)
            batch_x = scaler.transform(batch_x)
            '''
            batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)

            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)
            
            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            test_ims = np.stack(bothx_y,axis=0)   
            print("np.array(test_ims).shape=",np.array(test_ims).shape)

            test_dat = preprocess.reshape_patch(test_ims, args.patch_size)
            print("test_dat  preprocess.reshape_patch=",np.array(test_dat).shape)
            img_gen = model.test(test_dat, real_input_flag)
            print("--預測-")
            print("img_gen model.test=",np.array(img_gen).shape)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)
            output_length = args.total_length - args.input_length
            img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]

            np.savetxt(save_path+'radar_8_240210_img_outy.txt', batch_y.reshape(-1), delimiter=' ')

            test_x_6 = test_ims[:,model_parameter['period']:,:,:,:]
            test_mse = np.square(test_x_6 - img_out).sum()



            if batch_id ==1:
                test_ims_all = test_x_6
                img_out_all = img_out
            else:
                test_ims_all = np.concatenate((test_ims_all , test_x_6) ,axis = 0)#old nee
                img_out_all = np.concatenate((img_out_all , img_out) ,axis = 0)#old nee
            
            print("test_ims_all",np.array(test_ims_all).shape)
            print("img_out_all",np.array(img_out_all).shape) 


            mse_picture_avg = ((test_mse/1)/model_parameter['predict_period'])/(512*512)
            sum_mse_index = sum_mse_index + mse_picture_avg
            # MSE per frame
            save_path_single_location = save_path+'512x512_test_tw/'
            save_path_index = save_path_single_location + '512x512_PD_index_{}_test/'.format(index)#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            # for one_size in range(8):
            sum_mse =0
            avg_seq_p1_cost=0
            for i in range(output_length):
                x = test_ims[:, i + args.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # gx = np.maximum(gx, 1)
                # gx = np.minimum(gx, 6)

                # gx = np.minimum(gx, 60)
                print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (8, 512, 512, 1) gx= (8, 512, 512, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse
                sum_mse +=mse
                account_mse = sum_mse/1
                # seq_p1_cost = seq_p1_cost + account_mse

            avg_seq_p1_cost = ((sum_mse/1)/model_parameter['predict_period'])/(512*512)
            batch_cost = batch_cost + avg_seq_p1_cost

            for one_size in range(1):
                save_test_path = save_path_index + 'batch_{}/'.format(one_size)
                if not os.path.isdir(save_test_path):
                    os.makedirs(save_test_path)
                for i in range(output_length):
                    # print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (8, 512, 512, 1) gx= (8, 512, 512, 1)
                    vis_gx = np.array(img_out[one_size, i, :, :, :]).reshape(512,512)#t9~t18
                    vis_x = np.array(test_ims[one_size,  i + args.input_length, :, :, :]).reshape(512,512)#10,11,12,13,14,15,16,17,18,19
                    vis_test_x = np.array(test_ims[one_size,  i , :, :, :]).reshape(512,512)#10,11,12,13,14,15,16,17,18,19
                    
                    # vis_test_x = scaler.inverse_transform(vis_test_x.reshape(-1))
                    vis_test_x = vis_test_x.reshape(512,512)
                    # print("vis_gx.shape = ",vis_gx.shape)
                    # vis_gx = np.maximum(vis_gx, 1)
                    # vis_gx = vis_gx+3
                    vis_gx[vis_gx <= 1] = 0
                    # vis_gx = np.minimum(vis_gx, 70)                
                    visualized_area_with_map(vis_gx, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_pred_{}'.format(i), savepath=save_test_path)
                    visualized_area_with_map(vis_x, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_gt_{}'.format(i), savepath=save_test_path)
                    # visualized_area_with_map(vis_test_x, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_test_x_{}'.format(i), savepath=save_test_path)
                    
                    fn = save_test_path+'_sqe{}.txt'.format(i)
                    mse = np.square(vis_x - vis_gx).sum()
                    div_h_w = mse/(512*512)
                    # vis_x[vis_x > 1.] = 1.
                    # vis_x[vis_x < 0.] = 0.

                    # vis_gx[vis_gx > 1.] = 1.
                    # vis_gx[vis_gx < 0.] = 0.
                    # vis_x = np.maximum(vis_x, 0)
                    # vis_x = np.minimum(vis_x, 1)

                    # vis_gx = np.maximum(vis_gx, 0)
                    # vis_gx = np.minimum(vis_gx, 1)
                    vis_x = vis_x/65
                    vis_gx = vis_gx/65

                    real_frm = np.uint8(vis_x * 255)
                    pred_frm = np.uint8(vis_gx * 255)

                    # real_frm =vis_x 
                    # pred_frm = vis_gx 
                    # for b in range(configs.batch_size):
                    score, _ = compare_ssim(pred_frm, real_frm, full=True, multichannel=True)
                    ssim[i] += score
                    avg_ssim+=score
                    
                    with open(fn,'a') as file_obj:
                        file_obj.write("model_name " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
                    # file_obj.write("test day len(test_generator.step_per_epoch*32) =  " + str(test_generator.step_per_epoch*1)  + '\n' )
                        file_obj.write("test batch "+str(one_size) +" i = " + str(i) +"mse = "+str(div_h_w)   + '\n' )
                        file_obj.write("test batch "+str(one_size) +" i = " + str(i) +"SSIM = "+str(score)   + '\n' )
        avg_mse_p1 = sum_mse_index/num_of_batch_size
        avg_batch_cost =batch_cost/num_of_batch_size
        avg_ssim = (avg_ssim/model_parameter['predict_period'])/num_of_batch_size
        # avg_ssim = np.mean(ssim)
        print("mse=",img_mse)
        # avg_mse = avg_mse / (batch_id * 1)
        print('mse per seq: ' + str(avg_mse))
        
        fn = save_path_single_location + 'test_mse.txt'
        with open(fn,'a') as file_obj:
            file_obj.write("model_name " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            file_obj.write("test day len(test_generator.step_per_epoch*32) =  " + str(test_generator.step_per_epoch*1)  + '\n' )
            file_obj.write("test day num_of_batch_size =  " + str(num_of_batch_size)  + '\n' )
           
            for i in range(args.total_length - args.input_length):
                # print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * 1))#
                print("avg 512x512 mse seq[",i,"] =",(img_mse[i] / (batch_id * 1))/(512*512))
                print("avg 512x512 ssim seq[",i,"] =",(ssim[i]))

    #           file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
                file_obj.write("avg 512x512 mse seq[" + str(i) + '],test loss: ' + str((img_mse[i] / (batch_id * 1))/(512*512)) + '\n')  
                # file_obj.write("avg 512x512 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]) + '\n')  
                file_obj.write("avg 512x512_2 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]/batch_id * 1) + '\n')  
                
            file_obj.write("test avg_mse =  " + str(avg_batch_cost)  + '\n' )
            file_obj.write("test avg_mse_p1 =  " + str(avg_mse_p1)  + '\n' )
            file_obj.write("test avg_ssim =  " + str(avg_ssim)  + '\n' )

        test_y_csi = test_ims[:,model_parameter['period']:,:,:,:]
        vis_gx_csi = img_out#t9~t18
    csi_picture(img_out = img_out_all,test_ims= test_ims_all,save_path = save_path+'csi_{}/'.format(data_name),data_name=data_name)
def val(model, save_path, model_pkl, itr, val_radar_xy_shuffle = None): 
    print("===========val_wrapper===========")
    # model.eval()
    # print("model.training() = ",model.training())
    from pickle import load

    main_path=save_path
    val_day=save_path+'val_day/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # model.load(save_path, model_pkl)
    # model.load_state_dict(torch.load('params.pkl'))
    if val_radar_xy_shuffle is not None:
        places=['max places'] 
        print("all val_radar_xy_shuffle shape=",val_radar_xy_shuffle.shape)
        num_of_batch_size = len(val_radar_xy_shuffle)//args.batch_size
        print("val_radar_xy_shuffle num_of_batch_size=",num_of_batch_size)
    else:
        val_generator = radar.generator('val', batch_size=1, save_path = val_day)#args.batch_size)
        num_of_batch_size = val_generator.step_per_epoch#!-1
        print("range(val_generator.step_per_epoch)=",range(val_generator.step_per_epoch))
        places=['Sun_Moon_Lake'] 
        
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []
    place_len=len(places)

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    p20_cost =0
    avg_p20_cost=0
    sum_mse = 0
    p20_mse =0
    sum_xy=np.zeros(0)

    for place in places: 
        print("places=",places)
        # num_of_batch_size = val_generator.step_per_epoch-1
        batch_id = 0
        batch_cost = 0
        avg_batch_cost = 0
        sum_mse = 0
        sum_mse_index = 0
        smaple_p20_number = 0
        smaple_number=0
        avg_ssim = 0
        sum_ssim=0
        for index in range(num_of_batch_size):
            
            batch_id = batch_id + 1
            batch_x, batch_y = val_generator.generator_getClassifiedItems_3(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)
            # bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            # val_ims = np.stack(bothx_y,axis=0)   
            # print("np.array(val_ims).shape=",np.array(val_ims).shape)
            
            if val_radar_xy_shuffle is not None:
                train_xy = val_radar_xy_shuffle[index*args.batch_size:(index+1)*args.batch_size,:,:,:,:]#!
                print("index*args.batch_size=",index*args.batch_size,"to",(index+1)*args.batch_size)
                batch_x,batch_y = np.split(train_xy, 2, axis=1)
                print("batch_x=",batch_x.shape,"batch_y",batch_y.shape)
            # else:
            #     batch_x, batch_y = val_generator.generator_sample(index, place)#!
            # batch_x = batch_x.astype(np.float16)  
            # batch_y = batch_y.astype(np.float16)
            # batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)



            # '''
            # if batch_y.shape[0] is 0:
            #     print("batch_x is zero(<5).shape=",batch_x.shape)#(1, 6, 64, 64, 1)
            #     continue
            # else:
            #     print("batch_x.shape=",batch_x.shape)#(1, 6, 64, 64, 1)
            #     print("batch_y.shape=",batch_y.shape)#(1, 6, 64, 64, 1)
            #     bothx_y_temp = np.concatenate((batch_x, batch_y), axis=1)
            #     smaple_number+=1
            # if sum_xy.shape[0] is 0:#第一次
            #     print("creat sum_xy before =",sum_xy.shape)
            #     sum_xy = bothx_y_temp
            #     print("creat sum_xy after =",sum_xy.shape)
            #     continue

            # if sum_xy.shape[0] < 4 and sum_xy is not False:#第二次到31次
            #     # print("add sum_xy before =",sum_xy.shape)(31, 12, 64, 64, 1)
            #     sum_xy = np.concatenate((bothx_y_temp, sum_xy), axis=0)
            #     # print("add sum_xy after =",sum_xy.shape)(32, 12, 64, 64, 1)
            #     continue
            
            # # print("count 32 sum_xy.shape=",np.array(sum_xy).shape)
            # # val_ims = sum_xy
            
            # batch_x,batch_y = np.split(sum_xy, 2, axis=1)
            # scaler_path = main_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            # if not os.path.isdir(scaler_path):
            #     os.makedirs(scaler_path)
            # # scaler = load(open('min_max_scaler_4_240210.pkl', 'rb'))       
            # # srandard
            # scaler = load(open(scaler_path+'srandard_scaler_4_240210.pkl', 'rb'))       
            # # scaler = load(open('normalizer_scaler_4_240210.pkl', 'rb'))   
            # batch_x = np.array(batch_x).reshape(-1,64*64)
            # batch_x = scaler.transform(batch_x)
            # '''

            batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)

            
            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            val_ims = bothx_y
            val_dat = preprocess.reshape_patch(val_ims, args.patch_size)
            print("test_dat  preprocess.reshape_patch=",np.array(val_dat).shape)
            img_gen = model.test(val_dat, real_input_flag)
            print("--預測-")
            print("img_gen model.test=",np.array(img_gen).shape)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)
            output_length = args.total_length - args.input_length
            # img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]
            test_x_6 = val_ims[:, model_parameter['period']:, :, :, :]
            mse = np.square(test_x_6 - img_out).sum()
            mse_picture_avg = ((mse/args.batch_size)/model_parameter['predict_period'])/(512*512)
            sum_mse_index = sum_mse_index + mse_picture_avg
            # MSE per frame
            seq_p1_cost =0
            sum_mse =0
            avg_seq_p1_cost= 0
            for i in range(output_length):
                x = val_ims[:, i + args.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # gx = np.maximum(gx, 0)
                # gx = np.minimum(gx, 60)
                print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (4, 64, 64, 1) gx= (4, 64, 64, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse
                sum_mse +=mse
                account_mse = sum_mse/args.batch_size
                seq_p1_cost = seq_p1_cost + account_mse 
                print("batch_id=",batch_id,"i + args.input_length=",i + args.input_length,"account_mse=",account_mse)
            
                vis_x = x.copy()
                vis_gx = gx.copy()
                vis_x = vis_x/65
                vis_gx = vis_gx/65

                real_frm = np.uint8(vis_x * 255).reshape(512,512)
                pred_frm = np.uint8(vis_gx * 255).reshape(512,512)


                # for b in range(1):
                score, _ = compare_ssim(pred_frm, real_frm, full=True, multichannel=True)
                ssim[i] += score     
                sum_ssim+=score           
            
            avg_seq_p1_cost = ((sum_mse/args.batch_size)/model_parameter['predict_period'])/(512*512)

            batch_cost = batch_cost + avg_seq_p1_cost

        save_path_single_location = save_path+'64x64_val_tw/'
        save_path_index = save_path_single_location + '64x64_PD_index{}/'.format(index)#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        avg_ssim = (sum_ssim/model_parameter['predict_period'])/num_of_batch_size
        
        if not os.path.isdir(save_path_index):
            os.makedirs(save_path_index)
        avg_batch_cost =batch_cost/num_of_batch_size
        avg_mse_p1 = sum_mse_index/num_of_batch_size
        print('mse per seq: ' + str(avg_mse))    
        fn = save_path_single_location + 'val_mse_div20_.txt'
        with open(fn,'a') as file_obj:
            file_obj.write("-----val mse itr ="+str(itr)+"----- \n")
            file_obj.write("model_pkl " + model_pkl  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            file_obj.write("val day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
            file_obj.write("place" + str(place)  + '\n' )
            
            # for i in range(args.total_length - args.input_length):
            #     print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
            #     print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
            #    file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
            file_obj.write("val loss:" + str(avg_batch_cost) + '\n')     
            file_obj.write("val mse:" + str(avg_mse_p1) + '\n')  
            file_obj.write("val avg_ssim:" + str(avg_ssim) + '\n') 
            # file_obj.write("smaple_number:" + str(smaple_number) + '\n')  




        p20_mse = p20_mse+ avg_mse_p1
        p20_cost = p20_cost + avg_batch_cost
    p20_mse = p20_mse/place_len
    avg_p20_cost = p20_cost/place_len
    fn = save_path_single_location + 'val_mse_avg20_.txt'
    with open(fn,'a') as file_obj:
        file_obj.write("-----val mse itr ="+str(itr)+"----- \n")
        file_obj.write("model_pkl " + model_pkl  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        file_obj.write("val day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
        file_obj.write("place 20"   + '\n' )
        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
        #   file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write("val loss:" + str(avg_p20_cost) + '\n') 

        file_obj.write("p20_mse mse:" + str(p20_mse) + '\n')  


    fn = save_path_single_location + 'val_mse_itr.txt'
    with open(fn,'a') as file_obj:
        # file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
        # file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        # file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
        # file_obj.write("place 20"   + '\n' )
        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
        #    file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write(str(p20_mse) + '\n') 

    fn = save_path_single_location + 'val_ssim_itr.txt'
    with open(fn,'a') as file_obj:
        file_obj.write(str(avg_ssim) + '\n') 
    return p20_mse, avg_ssim
            
    #         print("mse=",img_mse)
    #         avg_mse = avg_mse / (batch_id * args.batch_size)

    #         print('mse per seq: ' + str(avg_mse))    
    #         fn = save_path_single_location + 'val_mse_div20.txt'
    #         with open(fn,'a') as file_obj:
    #             file_obj.write("-----val mse itr ="+str(itr)+"----- \n")
    #             file_obj.write("model_pkl " + model_pkl  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
    #             file_obj.write("val day len(num_of_batch_size*32) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
    #             for i in range(args.total_length - args.input_length):
    #                 print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
    #                 print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
    #     #           file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
    #                 file_obj.write("avg 64x64 mse seq[" + str(i) + '],val loss: ' + str(((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4) + '\n')          
    #             file_obj.write("avg 64x64 ,val loss: " + str(((avg_mse/6)/(64*64))/4) + '\n')


    #     fn = save_path_single_location + 'val_mse_nodiv20.txt'
    #     with open(fn,'a') as file_obj:
    #         file_obj.write("-----val mse itr ="+str(itr)+"----- \n")
    #         file_obj.write("model_pkl " + model_pkl  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
    #         file_obj.write("val day len(num_of_batch_size*32) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
    #         for i in range(args.total_length - args.input_length):
    #             print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
    #             print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64)))
    # #           file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
    #             file_obj.write("avg 64x64 mse seq[" + str(i) + '],val loss: ' + str((img_mse[i] / (batch_id * args.batch_size))/(64*64)) + '\n')          
    #         file_obj.write("avg 64x64 ,val loss: " + str(((avg_mse/6)/(64*64))) + '\n')

def tw(model, save_path, model_name,itr):
    #  model.load(args.pretrained_model)
    # date_date=[['2020-05-21 22:50', '2020-05-21 02:20']]
    # test_date=[['2020-05-21 00:10', '2020-05-21 00:11']]
    # date_date=[['2018-03-12 22:00', '2018-03-14 01:59']]
    
    # date_date=[['2018-06-14 22:00', '2018-06-16 01:59']]
    # test_date=[['2018-06-15 00:10', '2018-06-15 23:59']]
    date_date=[['2018-08-24 01:10', '2018-08-24 01:11']]
    test_date=[['2018-08-24 01:10', '2018-08-24 01:11']]
    # date_date=[['2018-06-15 00:10', '2018-06-15 00:11']]
    # test_date=[['2018-06-15 00:10', '2018-06-15 00:11']]
    # date_date=[['2020-05-27 00:00', '2020-05-27 00:01']]
    # test_date=[['2020-05-27 00:00', '2020-05-27 00:01']]

    # test_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
           #  ['2020-05-16 00:00','2020-05-16 23:59'],
            #  ['2020-05-19 00:00','2020-05-19 23:59'],
            #  ['2020-05-21 00:00','2020-05-21 23:59'],
            #  ['2020-05-22 00:00','2020-05-22 23:59'],
            #  ['2020-05-23 00:00','2020-05-23 23:59'],
            #  ['2020-05-26 00:00','2020-05-26 23:59'],
            #  ['2020-05-27 00:00','2020-05-27 23:59'],
            #  ['2020-05-28 00:00','2020-05-28 23:59'],
            #  ['2020-05-29 00:00','2020-05-29 23:59']]
    #places=['Bao_Zhong']
    places=['Sun_Moon_Lake']      
    # places=['Banqiao','Keelung']
    radar_echo_storage_path= 'D:/yu_ting/try/NWP/'#'NWP/'
    load_radar_echo_df_path = None
    #'data/RadarEcho_Sun_Moon_Lake_T6toT6_inoutputshape512.pkl'#'data/RadarEcho_64x64_2018_train8d_test824.pkl''data/RadarEcho_64x64_2018_0824.pkl'None
    radar_tw = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                    load_radar_echo_df_path=load_radar_echo_df_path,
                    input_shape=[512,512],#model_parameter['input_shape'],
                    output_shape=[512,512],#model_parameter['output_shape'],
                    period=model_parameter['period'],
                    predict_period=model_parameter['predict_period'],
                    places=places,
                    random=False,
                    date_range=date_date,
                    test_date=test_date)
    if not load_radar_echo_df_path:
        radar_tw.exportRadarEchoFileList()
        radar_tw.saveRadarEchoDataFrame()
        
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.load(save_path, model_name)
    save_path_tw = save_path+'320x320_test_itr{}_201808240000/'.format(itr)#!
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    test_generator = radar_tw.generator('test', batch_size=1)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (1,#args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    
    for place in places:
        #        ims = generator_getClassifiedItems( test_generator,  places=[place], all_list=None) 
        print("places=",places)
        print("range(test_generator.step_per_epoch)=",range(test_generator.step_per_epoch))
        num_of_batch_size = test_generator.step_per_epoch#-1
        for index in range(num_of_batch_size):
            batch_id = batch_id + 1 
            batch_x, batch_y = test_generator.generator_getClassifiedItems_3(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 320, 320, 1)
            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            test_ims = np.stack(bothx_y,axis=0)   
            print("np.array(test_ims).shape=",np.array(test_ims).shape)           
            # test_dat = preprocess.reshape_patch(test_ims, args.patch_size)
            # print("test_dat  preprocess.reshape_patch=",np.array(test_dat).shape)
            # np.array(test_ims).shape= (8, 12, 64, 64, 1)
            # reshape_patch
            # test_dat  preprocess.reshape_patch= (8, 12, 16, 16, 16)
            # period=model_parameter['period'],
                    # predict_period=model_parameter['predict_period']
            for i in range(5):#y
                print("=====i=",i,"=====")
                for j in range(5):#x
                    print("---j=",j,"---")
                    print("64*i:64*(i+1)=",64*i,"to",64*(i+1),"64*j:64*(j+1)=",64*j,"to",64*(j+1))
                    test_ims_block = test_ims[:,:,64*i:64*(i+1),64*j:64*(j+1),:]
                    # print("np.array(test_ims_block).shape=",np.array(test_ims_block).shape)#    (8, 12, 64, 64, 1)        
                    test_dat = preprocess.reshape_patch(test_ims_block, args.patch_size)
                    # print("test_dat  preprocess.reshape_patch=",np.array(test_dat).shape)#(8, 12, 16, 16, 16)
                    img_gen = model.test(test_dat, real_input_flag)
                    # print("img_gen model.test=",np.array(img_gen).shape)#(8, 11, 16, 16, 16)
                    print("預測")
                    # reshape_patch_back
                    img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
                    # print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)# (8, 11, 64, 64, 1)
                    output_length = args.total_length - args.input_length
                    # img_gen_length = img_gen.shape[1]
                    img_out = img_gen[:, -output_length:]
                    # print("img_out.shape=",np.array(img_out).shape)#img_out.shape= (8, 6, 64, 64, 1)

                    test_ims_block = test_ims_block.reshape(1, model_parameter['period']+model_parameter['predict_period'], 64, 64)
                    img_gen_block = img_gen.reshape(1,model_parameter['period']+model_parameter['predict_period']-1,64,64)
                    # print("test_ims_block.shape = ",test_ims_block.shape)    
                    if j == 0:
                        test_ims_block_hstack=test_ims_block
                        # print("j 0 test_ims_block_hstack.shape =>",test_ims_block_hstack.shape)#(8, 12, 64, 64)
                        img_gen_block_hstack=img_gen_block
                        print("j 0 img_gen_block_hstack.shape =>",img_gen_block_hstack.shape)

                    else:
                        # print("i = ",i," j = ",j," test_ims_block_hstack.shape = ",test_ims_block_hstack.shape,"test_ims_block.shape = ",test_ims_block.shape) 
                        test_ims_block_hstack=np.concatenate((test_ims_block_hstack,test_ims_block),axis=3)#old new 向右連接
                        #! ifi =  4  j =  1  test_ims_block_hstack.shape =  (8, 12, 64, 64) test_ims_block.shape =  (8, 12, 64, 64)
                        
                        print("i = ",i," j = ",j," img_gen_block_hstack.shape = ",img_gen_block_hstack.shape,"img_gen_block.shape = ",img_gen_block.shape) 
                        img_gen_block_hstack=np.concatenate((img_gen_block_hstack,img_gen_block),axis=3)#old new 向右連接
                        print("after concatenate img_gen_block_hstack.shape=",img_gen_block_hstack.shape)

                if i == 0:
                    test_ims_block_vstack = test_ims_block_hstack #!第一次要堆疊
                    print("i 0 test_ims_block_vstack=",test_ims_block_vstack.shape)#i 0 test_ims_block_vstack= (8, 12, 64, 320)
                    
                    img_gen_block_vstack = img_gen_block_hstack
                    print("i 0 img_gen_block_vstack=",img_gen_block_vstack.shape)#

                else:
                    print("test_ims_block_vstack=",test_ims_block_vstack.shape,"test_ims_block_hstack.shape=",test_ims_block_hstack.shape)
                    test_ims_block_vstack = np.concatenate((test_ims_block_vstack , test_ims_block_hstack) ,axis = 2)#old nee 向上連接
                    # i =  4  j =  4  test_ims_block_hstack.shape =  (8, 12, 64, 256) test_ims_block.shape =  (8, 12, 64, 64)
                    # test_ims_block_vstack= (8, 12, 256, 320) test_ims_block_hstack.shape= (8, 12, 64, 320)
                    # #! if i&j == 4 than test_ims_block_vstack(GT) (8, 12, 256, 320)+(8, 12, 64, 320)=>(8, 12, 320, 320)
                    img_gen_block_vstack = np.concatenate((img_gen_block_vstack , img_gen_block_hstack) ,axis = 2)#old nee 向上連接
                    print("after concatenate img_gen_block_vstack.shape=",img_gen_block_vstack.shape)
                    
                #            
                print("i = ",i," j = ",j," test_ims_block_vstack.shape = ",test_ims_block_vstack.shape) 
                # i =  4  j =  4  test_ims_block_vstack.shape =  (8, 12, 320, 320)
                print("i = ",i," j = ",j," img_gen_block_vstack.shape = ",img_gen_block_vstack.shape) 

            # sys.exit()
                # visualized_radar_picture(test_ims_block, img_out, save_path, num_of_batch_size)

            output_seq=["010","020","030","040","050","060"]
            time_seq = ['0000','0010',"0020","0030","0040","0050","0100","0110"]
            # save_path = 'save_3mitr3_0912/315x315_test_tw/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_test
            #    for index in range(num_of_batch_size):
            # '''
            save_path = save_path_tw + '320x320_GT_index{}/'.format(index)#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            for one_of_batch in range(1):#batch_size     
                path = save_path + '{}/'.format(one_of_batch)
                if not os.path.isdir(path):
                    os.mkdir(path) 
                for one_of_time in range(test_ims_block_vstack.shape[1]):#6):#gt_range_temp[index]
                    # name = 'gt' + '{}'.format(i+6)
                    name = 'gt' + '{}'.format(one_of_time)
                    # file_name = os.path.join(path, name)

                    # img_gt =test_ims[one_of_batch, one_of_time, :, :, :].reshape(64,64)
                    # visualized_area_with_map(img_gt, 'Bao_Zhong', shape_size=[64,64], title=name, savepath=path)
                    test_ims_one = test_ims_block_vstack[one_of_batch, one_of_time, :, :].reshape(320,320)
                    visualized_area_with_map(test_ims_one, 'Sun_Moon_Lake', shape_size=[320,320], title=name, savepath=path)
            
            # for index in range(num_of_batch_size):
            
            save_path = save_path_tw + '320x320_PD_index{}_img_gen_one_small_1_do_s0/'.format(index)#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            for one_of_batch in range(1):#8):#batch_size     
                path = save_path + '{}/'.format(one_of_batch)
                if not os.path.isdir(path):
                    os.mkdir(path) 
                for one_of_time in range(img_gen_block_vstack.shape[1]):#6):#gt_range_temp[index]
                    # name = 'gt' + '{}'.format(i+6)
                    name = 'pd' + '{}'.format(one_of_time)
                    # file_name = os.path.join(path, name)

                    # img_gt =test_ims[one_of_batch, one_of_time, :, :, :].reshape(64,64)
                    # visualized_area_with_map(img_gt, 'Bao_Zhong', shape_size=[64,64], title=name, savepath=path)
                    img_gen_one = img_gen_block_vstack[one_of_batch, one_of_time, :, :].reshape(320,320)
                    test_ims_one_ = test_ims_block_vstack[one_of_batch, one_of_time, :, :].reshape(320,320)

                    # img_gen_one = np.minimum(img_gen_one, 10)
                    fn = path+'_sqe{}.txt'.format(one_of_time)
                    mse = np.square(test_ims_one_ - img_gen_one).sum()
                    div_h_w = mse/(320*320)
                    with open(fn,'a') as file_obj:
                        file_obj.write("model_name " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
                    # file_obj.write("test day len(test_generator.step_per_epoch*32) =  " + str(test_generator.step_per_epoch*1)  + '\n' )
                        file_obj.write("test batch "+str(one_of_batch) +" one_of_time = " + str(one_of_time) +"mse = "+str(div_h_w)   + '\n' )
                    
                    img_gen_one[img_gen_one < 1] = 0
                    visualized_area_with_map(img_gen_one, 'Sun_Moon_Lake', shape_size=[320,320], title=name, savepath=path)
def load_sample(save_path):
    
    from sklearn.utils import shuffle
    
    save_np=save_path+'sum_sample/'#'sample_radar/'
    if not os.path.isdir(save_np):
        os.makedirs(save_np)

    max40_x_Hualien = np.load(save_np + '2018_601to830_radarmax40_x_Hualien.npy')
    max40_y_Hualien = np.load(save_np + '2018_601to830_radarmax40_y_Hualien.npy')
    # avg5_x_Hualien = np.load(save_np + '2018_601to830_radaravg5_x_Hualien.npy')
    # avg5_y_Hualien = np.load(save_np + '2018_601to830_radaravg5_y_Hualien.npy')


    
    max40_x_Banqiao = np.load(save_np + '2018_601to830_radarmax40_x_Banqiao.npy')
    max40_y_Banqiao = np.load(save_np + '2018_601to830_radarmax40_y_Banqiao.npy')
    # avg5_x_Banqiao = np.load(save_np + '2018_601to830_radaravg5_x_Banqiao.npy')
    # avg5_y_Banqiao = np.load(save_np + '2018_601to830_radaravg5_y_Banqiao.npy')

    max40_x_Sun_Moon_Lake = np.load(save_np + '2018_601to830_radarmax40_x_Sun_Moon_Lake.npy')
    max40_y_Sun_Moon_Lake = np.load(save_np + '2018_601to830_radarmax40_y_Sun_Moon_Lake.npy')
    # avg5_x_Sun_Moon_Lake = np.load(save_np + '2018_601to830_radaravg5_x_Sun_Moon_Lake.npy')
    # avg5_y_Sun_Moon_Lake = np.load(save_np + '2018_601to830_radaravg5_y_Sun_Moon_Lake.npy')



    max40_x_Bao_Zhong = np.load(save_np + '2018_601to830_radarmax40_x_Bao_Zhong.npy')
    max40_y_Bao_Zhong = np.load(save_np + '2018_601to830_radarmax40_y_Bao_Zhong.npy')
    # avg5_x_Bao_Zhong = np.load(save_np + '2018_601to830_radaravg5_x_Bao_Zhong.npy')
    # avg5_y_Bao_Zhong = np.load(save_np + '2018_601to830_radaravg5_y_Bao_Zhong.npy')


    # max40_x_Shangdewen = np.load(save_np + '2018_601to830_radarmax30_x_Shangdewen.npy')
    # max40_y_Shangdewen = np.load(save_np + '2018_601to830_radarmax30_y_Shangdewen.npy')
    # avg5_x_Shangdewen = np.load(save_np + '2018_601to830_radaravg5_x_Shangdewen.npy')
    # avg5_y_Shangdewen = np.load(save_np + '2018_601to830_radaravg5_y_Shangdewen.npy')


    # max40_x_Keelung= np.load(save_np + '2018_601to830_radarmax30_x_Keelung.npy')
    # max40_y_Keelung= np.load(save_np + '2018_601to830_radarmax30_y_Keelung.npy')
    # avg5_x_Keelung= np.load(save_np + '2018_601to830_radaravg5_x_Keelung.npy')
    # avg5_y_Keelung= np.load(save_np + '2018_601to830_radaravg5_y_Keelung.npy')

    max40_y_Hualien = max40_y_Hualien.reshape(-1,model_parameter['predict_period'],64,64,1)
    # avg5_y_Hualien = avg5_y_Hualien.reshape(-1,model_parameter['predict_period'],64,64,1)
    max40_y_Banqiao = max40_y_Banqiao.reshape(-1,model_parameter['predict_period'],64,64,1)
    # avg5_y_Banqiao = avg5_y_Banqiao.reshape(-1,model_parameter['predict_period'],64,64,1)
    max40_y_Sun_Moon_Lake = max40_y_Sun_Moon_Lake.reshape(-1,model_parameter['predict_period'],64,64,1)
    # avg5_y_Sun_Moon_Lake = avg5_y_Sun_Moon_Lake.reshape(-1,model_parameter['predict_period'],64,64,1)
    max40_y_Bao_Zhong = max40_y_Bao_Zhong.reshape(-1,model_parameter['predict_period'],64,64,1)
    # avg5_y_Bao_Zhong = avg5_y_Bao_Zhong.reshape(-1,model_parameter['predict_period'],64,64,1)
    # max40_y_Shangdewen = max40_y_Shangdewen.reshape(-1,model_parameter['predict_period'],64,64,1)
    # avg5_y_Shangdewen = avg5_y_Shangdewen.reshape(-1,model_parameter['predict_period'],64,64,1)


    # max40_y_Keelung = max40_y_Keelung.reshape(-1,model_parameter['predict_period'],64,64,1)
    # avg5_y_Keelung = avg5_y_Keelung.reshape(-1,model_parameter['predict_period'],64,64,1)

    # Shangdewen
    # Keelung
    print("max40_x_Hualien,max40_y_Hualien",max40_x_Hualien.shape,max40_y_Hualien.shape)
    # print("avg5_x_Hualien, avg5_y_Hualien",avg5_x_Hualien.shape , avg5_y_Hualien.shape)
    print("max40_x_Banqiao, max40_y_Banqiao", max40_x_Banqiao.shape , max40_y_Banqiao.shape)
    # print("avg5_x_Banqiao, avg5_y_Banqiao",avg5_x_Banqiao.shape , avg5_y_Banqiao.shape)
    print("max40_x_Sun_Moon_Lake, max40_y_Sun_Moon_Lake",max40_x_Sun_Moon_Lake.shape , max40_y_Sun_Moon_Lake.shape)
    # print("avg5_x_Sun_Moon_Lake, avg5_y_Sun_Moon_Lake",avg5_x_Sun_Moon_Lake.shape , avg5_y_Sun_Moon_Lake.shape)
    print("max40_x_Bao_Zhong, max40_y_Bao_Zhong",max40_x_Bao_Zhong.shape , max40_y_Bao_Zhong.shape)
    # print("avg5_x_Bao_Zhong, avg5_y_Bao_Zhong",avg5_x_Bao_Zhong.shape , avg5_y_Bao_Zhong.shape)


    # print("max40_x_Shangdewen, max40_y_Shangdewen",max40_x_Shangdewen.shape , max40_y_Shangdewen.shape)
    # print("avg5_x_Shangdewen, avg5_y_Shangdewen",avg5_x_Shangdewen.shape , avg5_y_Shangdewen.shape)

    # print("max40_x_Keelung, max40_y_Keelung",max40_x_Keelung.shape , max40_y_Keelung.shape)
    # print("avg5_x_Keelung, avg5_y_Keelung",avg5_x_Keelung.shape , avg5_y_Keelung.shape)

    max40_Hualien_xy = np.concatenate((max40_x_Hualien , max40_y_Hualien) ,axis = 1)
    max40_Banqiao_xy = np.concatenate((max40_x_Banqiao , max40_y_Banqiao) ,axis = 1)
    max40_Sun_Moon_Lake_xy = np.concatenate((max40_x_Sun_Moon_Lake , max40_y_Sun_Moon_Lake) ,axis = 1)
    max40_Bao_Zhong_xy = np.concatenate((max40_x_Bao_Zhong , max40_y_Bao_Zhong) ,axis = 1)
    # max40_Shangdewen_xy = np.concatenate((max40_x_Shangdewen , max40_y_Shangdewen) ,axis = 1)
    # max40_Keelung_xy = np.concatenate((max40_x_Keelung , max40_y_Keelung) ,axis = 1)


    # avg5_Hualien_xy = np.concatenate((avg5_x_Hualien , avg5_y_Hualien) ,axis = 1)
    # avg5_Banqiao_xy = np.concatenate((avg5_x_Banqiao , avg5_y_Banqiao) ,axis = 1)
    # avg5_Sun_Moon_Lake_xy = np.concatenate((avg5_x_Sun_Moon_Lake , avg5_y_Sun_Moon_Lake) ,axis = 1)
    # avg5_Bao_Zhong_xy = np.concatenate((avg5_x_Bao_Zhong , avg5_y_Bao_Zhong) ,axis = 1)
    # avg5_Shangdewen_xy = np.concatenate((avg5_x_Shangdewen , avg5_y_Shangdewen) ,axis = 1)
    # avg5_Keelung_xy = np.concatenate((avg5_x_Keelung , avg5_y_Keelung) ,axis = 1)


    # train_radar_xy = np.concatenate((max40_Hualien_xy , max40_Banqiao_xy,max40_Sun_Moon_Lake_xy,max40_Bao_Zhong_xy, avg5_Hualien_xy, avg5_Banqiao_xy,avg5_Sun_Moon_Lake_xy,avg5_Bao_Zhong_xy,max40_Shangdewen_xy,avg5_Shangdewen_xy,max40_Keelung_xy,avg5_Keelung_xy) ,axis = 0)#!
    train_radar_xy = np.concatenate((max40_Hualien_xy, max40_Banqiao_xy, max40_Sun_Moon_Lake_xy, max40_Bao_Zhong_xy) ,axis = 0)#,max40_Shangdewen_xy,max40_Keelung_xy) ,axis = 0)#!


    '''
    radar0_x = np.load(save_np + 'radar0_x.npy')
    radar0_y = np.load(save_np + 'radar0_y.npy')
    radar1to9_x = np.load(save_np + 'radar1to9_x.npy')
    radar1to9_y = np.load(save_np + 'radar1to9_y.npy')
    radar10to19_x = np.load(save_np + 'radar10to19_x.npy')
    radar10to19_y = np.load(save_np + 'radar10to19_y.npy')
    radar20to29_x = np.load(save_np + 'radar20to29_x.npy')
    radar20to29_y = np.load(save_np + 'radar20to29_y.npy')
    
    # radar40to39_x = np.load(save_np + 'radar30to39_x.npy')
    # radar30to39_y = np.load(save_np + 'radar30to39_y.npy')
    # radar30to39_y = radar30to39_y.reshape(-1,model_parameter['predict_period'],64,64,1)
    radar40_x = np.load(save_np + 'radar40_x.npy')
    radar40_y = np.load(save_np + 'radar40_y.npy')
    radar0_y = radar0_y.reshape(-1,model_parameter['predict_period'],64,64,1)
    radar1to9_y = radar1to9_y.reshape(-1,model_parameter['predict_period'],64,64,1)
    radar10to19_y = radar10to19_y.reshape(-1,model_parameter['predict_period'],64,64,1)
    radar20to29_y = radar20to29_y.reshape(-1,model_parameter['predict_period'],64,64,1)

    radar40_y = radar40_y.reshape(-1,model_parameter['predict_period'],64,64,1)

    
    print("radar0_x.shape",radar0_x.shape,radar0_y.shape)
    print("radar1to9_x, radar1to9_y",radar1to9_x.shape , radar1to9_y.shape)
    print(" radar10to19_x, radar10to19_y", radar10to19_x.shape , radar10to19_y.shape)
    print("radar20to29_x, radar20to29_y",radar20to29_x.shape , radar20to29_y.shape)
    
    # print("radar30to39_x, radar30to39_y",radar30to39_x.shape, radar30to39_y.shape)
    print("radar40_x, radar40_y",radar40_x.shape, radar40_y.shape)
    
    train_radar0 = np.concatenate((radar0_x , radar0_y) ,axis = 1)
    train_radar1to9 = np.concatenate((radar1to9_x , radar1to9_y) ,axis = 1)
    train_radar10to19 = np.concatenate((radar10to19_x , radar10to19_y) ,axis = 1)
    train_radar20to29 = np.concatenate((radar20to29_x , radar20to29_y) ,axis = 1)
    
    # train_radar30to39 = np.concatenate((radar30to39_x , radar30to39_y) ,axis = 1)
    train_radar40 = np.concatenate((radar40_x , radar40_y) ,axis = 1)
    train_radar0 = train_radar0[:1000,:,:,:,:]
    train_radar1to9 = train_radar1to9#[:1000,:,:,:,:]
    train_radar10to19 = train_radar10to19#[:1000,:,:,:,:]
    train_radar20to29 = train_radar20to29#[:1000,:,:,:,:]

    
    
    
    # train_radar30to39 = train_radar30to39[:1000,:,:,:,:]
    # print("1000 train_radar30to39.shape=",train_radar30to39.shape)

    train_radar40 = train_radar40#[:1000,:,:,:,:]
    print("1000 train_radar40.shape=",train_radar40.shape)

    # train_radar_xy = np.concatenate((train_radar30to39, train_radar40) ,axis = 0)
    # train_radar_xy = np.concatenate((train_radar0 , train_radar1to9, train_radar10to19, train_radar20to29, train_radar30to39, train_radar40) ,axis = 0)#!
    train_radar_xy = np.concatenate((train_radar0 , train_radar1to9, train_radar10to19, train_radar20to29, train_radar40) ,axis = 0)#!
    '''

 
    print("train_radar_xy.shape=",train_radar_xy.shape)
    train_radar_xy_shuffle = shuffle(train_radar_xy)
    val_radar_xy_shuffle = train_radar_xy_shuffle[:int(0.2*len(train_radar_xy_shuffle)),:,:,:,:]
    train_radar_xy_shuffle = train_radar_xy_shuffle[int(0.2*len(train_radar_xy_shuffle)):,:,:,:,:] 
    print("val_radar_xy_shuffle.shape=",val_radar_xy_shuffle.shape)
    print("train_radar_xy_shuffle.shape=",train_radar_xy_shuffle.shape)
    # train_radar_xy_shuffle =train_radar_xy_shuffle[:1000,:,:,:,:]
    fn = save_np + 'radar_sample_number.txt'
    with open(fn,'a') as file_obj:
        '''
        file_obj.write("len(train_radar0) = "+str(len(train_radar0))+"\n")
        file_obj.write("len(train_radar1to9) = "+str(len(train_radar1to9))+"\n")
        file_obj.write("len(train_radar10to19) = "+str(len(train_radar10to19))+"\n")
        file_obj.write("len(train_radar20to29) = "+str(len(train_radar20to29))+"\n")
        # file_obj.write("len(train_radar30to39)"+str(len(train_radar30to39))+"\n")

        file_obj.write("len(train_radar40) = "+str(len(train_radar40))+"\n")
        '''
        file_obj.write("max40_Hualien_xy = "+str(max40_Hualien_xy.shape)+"\n")
        file_obj.write("max40_Banqiao_xy = "+str(max40_Banqiao_xy.shape)+"\n")
        file_obj.write("max40_Sun_Moon_Lake_xy = "+str(max40_Sun_Moon_Lake_xy.shape)+"\n")
        
        file_obj.write("max40_Banqiao_xy = "+str(max40_Banqiao_xy.shape)+"\n")

        # file_obj.write("avg5_Hualien_xy = "+str(avg5_Hualien_xy.shape)+"\n")
        # file_obj.write("avg5_Banqiao_xy = "+str(avg5_Banqiao_xy.shape)+"\n")
        # file_obj.write("avg5_Sun_Moon_Lake_xy = "+str(avg5_Sun_Moon_Lake_xy.shape)+"\n")


        
        file_obj.write("len(train_radar_xy_shuffle) = "+str(train_radar_xy_shuffle.shape)+"\n")

        file_obj.write("len(val_radar_xy_shuffle) = "+str(val_radar_xy_shuffle.shape)+"\n")

    np.save(save_np + 'train_radar_xy_shuffle_p4_20180601to0830_max40.npy',train_radar_xy_shuffle)
    np.save(save_np + 'val_radar_xy_shuffle_p4_20180601to0830_max40.npy',val_radar_xy_shuffle)



    return train_radar_xy_shuffle, val_radar_xy_shuffle
def muti_sample(save_path, type='MinMaxScaler'): 
    from sklearn import preprocessing
    from pickle import dump
    print(save_path)      
    train_day_path = save_path+'train_day/'
    train_generator = radar.generator('train', batch_size=1024,save_path=train_day_path)
    num_of_batch_size = train_generator.step_per_epoch#!-1
 
    sample_p20 =[]
    place_len=len(places)
    print("place_len = ",place_len)
    print("num_of_batch_size = ",num_of_batch_size)
    all_batch_x = []
    all_batch_y = []
    one = True
    radarmax20_x=[]
    radaravg5_x=[]

    radarmax20_y=[]
    radaravg5_y=[]
    for place in places:
        for index in range(num_of_batch_size):
            # radar0_x, radar0_y, radar1to9_x, radar1to9_y, radar10to19_x, radar10to19_y, radar20to29_x, radar20to29_y, radar30to39_x, radar30to39_y, radar40_x, radar40_y = train_generator.muti_sample(index, place)
            radarmax20_x, radarmax20_y, radaravg5_x, radaravg5_y = train_generator.generator_max_sample(index, place)
            print("num_of_batch_size=",num_of_batch_size,"index=",index)
            
            print(str(place),"radarmax20_x",radarmax20_x.shape)
            print(str(place),"radarmax20_y",radarmax20_y.shape)
            # print(str(place),"radarmax20_y type",type(radarmax20_y))


            print(str(place),"radaravg5_x",radaravg5_x.shape)
            print(str(place),"radaravg5_y",radaravg5_y.shape)

            # sample10to29 = radar10to19_x.shape[0]+radar20to29_x.shape[0]
            '''
            if radar1to9_x.shape[0]>1000 and radar40_x.shape[0]>1000:
                save_np=save_path+'sample_radar/'
                if not os.path.isdir(save_np):
                    os.makedirs(save_np)
                np.save(save_np + 'radar0_x.npy', radar0_x)
                np.save(save_np + 'radar0_y.npy', radar0_y)
                np.save(save_np + 'radar1to9_x.npy', radar1to9_x)
                np.save(save_np + 'radar1to9_y.npy', radar1to9_y)
                np.save(save_np + 'radar10to19_x.npy', radar10to19_x)
                np.save(save_np + 'radar10to19_y.npy', radar10to19_y)
                np.save(save_np + 'radar20to29_x.npy', radar20to29_x)
                np.save(save_np + 'radar20to29_y.npy', radar20to29_y)

                # np.save(save_np + 'radar30to39_x.npy', radar30to39_x)
                np.save(save_np + 'radar30to39_y.npy', radar30to39_y)
                np.save(save_np + 'radar40_x.npy', radar40_x)
                np.save(save_np + 'radar40_y.npy', radar40_y)        
                print("radar0_x.shape",radar0_x.shape,radar0_y.shape)
                print("radar1to9_x, radar1to9_y",radar1to9_x.shape , radar1to9_y.shape)
                print(" radar10to19_x, radar10to19_y", radar10to19_x.shape , radar10to19_y.shape)
                print("radar20to29_x, radar20to29_y",radar20to29_x.shape , radar20to29_y.shape)
                print("radar30to39_x, radar30to39_y",radar30to39_x.shape, radar30to39_y.shape)
                print("radar40_x, radar40_y",radar40_x.shape, radar40_y.shape)

                np.save(save_np + 'radar0_x{}.npy'.format(len(radar0_x)), radar0_x)
                np.save(save_np + 'radar0_y{}.npy'.format(len(radar0_y)), radar0_y)
                np.save(save_np + 'radar1to9_x{}.npy'.format(len(radar1to9_x)), radar1to9_x)
                np.save(save_np + 'radar1to9_y{}.npy'.format(len(radar1to9_y)), radar1to9_y)
                np.save(save_np + 'radar10to19_x{}.npy'.format(len(radar10to19_x)), radar10to19_x)
                np.save(save_np + 'radar10to19_y{}.npy'.format(len(radar10to19_y)), radar10to19_y)
                np.save(save_np + 'radar20to29_x{}.npy'.format(len(radar20to29_x)), radar20to29_x)
                np.save(save_np + 'radar20to29_y{}.npy'.format(len(radar20to29_y)), radar20to29_y)

                # np.save(save_np + 'radar30to39_x{}.npy'.format(len(radar30to39_x)), radar30to39_x)
                np.save(save_np + 'radar30to39_y{}.npy'.format(len(radar30to39_y)), radar30to39_y)
                np.save(save_np + 'radar40_x{}.npy'.format(len(radar40_x)), radar40_x)
                np.save(save_np + 'radar40_y{}.npy'.format(len(radar40_y)), radar40_y)
                return 0
            '''
    # if radar1to9_x.shape[0]>1000 and radar40_x.shape[0]>1000:
        save_np=save_path+'sample_radar_{}_2018_601to830/'.format(place)
        if not os.path.isdir(save_np):
            os.makedirs(save_np)
        np.save(save_np + '2018_601to830_radarmax40_x_{}.npy'.format(place), radarmax20_x)
        np.save(save_np + '2018_601to830_radarmax40_y_{}.npy'.format(place), radarmax20_y)
        np.save(save_np + '2018_601to830_radaravg5_x_{}.npy'.format(place), radaravg5_x)
        np.save(save_np + '2018_601to830_radaravg5_y_{}.npy'.format(place), radaravg5_y)



    '''
    radar0_x = np.load(save_np + 'radar0_x.npy')
    radar0_y = np.load(save_np + 'radar0_y.npy')
    radar1to9_x = np.load(save_np + 'radar1to9_x.npy')
    radar1to9_y = np.load(save_np + 'radar1to9_y.npy')
    radar10to19_x = np.load(save_np + 'radar10to19_x.npy')
    radar10to19_y = np.load(save_np + 'radar10to19_y.npy')
    radar20to29_x = np.load(save_np + 'radar20to29_x.npy')
    radar20to29_y = np.load(save_np + 'radar20to29_y.npy')

    radar30to39_x = np.load(save_np + 'radar30to39_x.npy')
    radar30to39_y = np.load(save_np + 'radar30to39_y.npy')
    radar40_x = np.load(save_np + 'radar40_x.npy')
    radar40_y = np.load(save_np + 'radar40_y.npy')

    print("radar0_x.shape",radar0_x.shape,radar0_y.shape)
    print("radar1to9_x, radar1to9_y",radar1to9_x.shape , radar1to9_y.shape)
    print(" radar10to19_x, radar10to19_y", radar10to19_x.shape , radar10to19_y.shape)
    print("radar20to29_x, radar20to29_y",radar20to29_x.shape , radar20to29_y.shape)
    print("radar30to39_x, radar30to39_y",radar30to39_x.shape, radar30to39_y.shape)
    print("radar40_x, radar40_y",radar40_x.shape, radar40_y.shape)
    '''

def preprocess_sample(save_path, type='MinMaxScaler'): 
    from sklearn import preprocessing
    from pickle import dump      
    train_day_path = save_path+'train_day/'
    train_generator = radar.generator('train', batch_size=1,save_path=train_day_path)
    num_of_batch_size = train_generator.step_per_epoch#!-1
 
    sample_p20 =[]
    place_len=len(places)
    print("place_len = ",place_len)
    print("num_of_batch_size = ",num_of_batch_size)
    all_batch_x = []
    all_batch_y = []
    one = True
    for place in places:
        for index in range(num_of_batch_size):
            # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
            batch_x, batch_y = train_generator.generator_sample(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            # all_batch_x.append(batch_x)

            if batch_y.shape[0] is 0:
                print("batch_x is zero(<5).shape=",batch_x.shape)#(1, 6, 64, 64, 1)
                continue
            else:
                batch_y = batch_y.reshape(len(batch_y), 6, 64, 64, 1)
                print("batch_x.shape=",batch_x.shape)#(1, 6, 64, 64, 1)
                print("batch_y.shape=",batch_y.shape)#(1, 6, 64, 64, 1)
                bothx_y = np.concatenate((batch_x, batch_y), axis=1)

            if one:
                all_batch_x = batch_x
                all_batch_y = batch_y
                one =False
            else:
                all_batch_x=np.concatenate((all_batch_x,batch_x),axis=0)
                all_batch_y=np.concatenate((all_batch_y,batch_x),axis=0)

    # print("all_batch_x",all_batch_x.shape) 
    
    # all_batch_x,batch_y = np.split(all_batch_x, 2, axis=1)

    print("split all_batch_x",all_batch_x.shape) 
    all_batch_x = all_batch_x.reshape(-1)
    print("all_batch_x.reshape(-1)",all_batch_x.shape)
    all_batch_y = all_batch_y.reshape(-1)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # np.savetxt(save_path+'radar_8_240210_smaple.txt', all_batch_x, delimiter=' ')
    np.save(save_path+'all_batch_x_sample', all_batch_x)
    np.save(save_path+'all_batch_y_smaple', all_batch_y)

    if type=='StandardScaler':
        save_path = save_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # scaler = preprocessing.StandardScaler().fit(X)
        srandard_scaler = preprocessing.StandardScaler()
        all_batch_x = all_batch_x.reshape(-1, 64*64)

        x_train_srandard = srandard_scaler.fit_transform(all_batch_x)
        dump(srandard_scaler, open(save_path+'srandard_scaler_8_240210.pkl', 'wb'))
        x_train_srandard = x_train_srandard.reshape(-1)
        fn = save_path + 'srandard_scaler_8_240210_trainx.txt'
        # np.savetxt(save_path+'srandard_scaler_8_240210_trainx.txt', x_train_srandard , delimiter=' ')
        np.save(save_path+'x_train_srandard_sample', x_train_srandard)

    # sys.exit() 
            #     batch_x, batch_y = val_generator.generator_sample(index, place)
            # batch_x = batch_x.astype(np.float16)  
            # batch_y = batch_y.astype(np.float16)
            # batch_y = batch_y.reshape(len(batch_y), 6, 64, 64, 1)
            # if batch_y.shape[0] is 0:
            #     print("batch_x is zero(<5).shape=",batch_x.shape)#(1, 6, 64, 64, 1)
            #     continue
            # else:
            #     print("batch_x.shape=",batch_x.shape)#(1, 6, 64, 64, 1)
            #     print("batch_y.shape=",batch_y.shape)#(1, 6, 64, 64, 1)
            #     bothx_y = np.concatenate((batch_x, batch_y), axis=1)
            #     smaple_number+=1
            # if sum_xy.shape[0] is 0:#第一次
            #     print("creat sum_xy before =",sum_xy.shape)
            #     sum_xy = bothx_y
            #     print("creat sum_xy after =",sum_xy.shape)
            #     continue

            # if sum_xy.shape[0] < 32 and sum_xy is not False:#第二次到31次
            #     # print("add sum_xy before =",sum_xy.shape)(31, 12, 64, 64, 1)
            #     sum_xy = np.concatenate((bothx_y, sum_xy), axis=0)
            #     # print("add sum_xy after =",sum_xy.shape)(32, 12, 64, 64, 1)
            #     continue
        
            # print("count 32 sum_xy.shape=",np.array(sum_xy).shape)
            # val_ims = sum_xy
def preprocess_fun(save_path, type='MinMaxScaler'):
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.preprocessing import Normalizer
    from sklearn import preprocessing
    from pickle import dump
    train_generator = radar.generator('train', batch_size=1)
    test_generator = radar.generator('test', batch_size=1)

    print("places=",places)
    print("range(train_generator.step_per_epoch)=",range(train_generator.step_per_epoch))
    print("range(test_generator.step_per_epoch)=",range(test_generator.step_per_epoch))

    # num_of_batch_size = train_generator.step_per_epoch#!-1
    num_of_batch_size = test_generator.step_per_epoch#!-1
 
    sample_p20 =[]
    place_len=len(places)
    print("place_len = ",place_len)
    print("num_of_batch_size = ",num_of_batch_size)
    all_batch_x = []
    one = True
    for place in places:
        for index in range(num_of_batch_size):
            # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
            batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            # all_batch_x.append(batch_x)
            if one:
                all_batch_x = batch_x
                one =False
            else:
                all_batch_x=np.concatenate((all_batch_x,batch_x),axis=0)
    print("all_batch_x",all_batch_x.shape)
    # all_batch_x.reshape(-)
    all_batch_x = all_batch_x.reshape(-1)
    print("all_batch_x",all_batch_x.shape)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    np.savetxt(save_path+'radar_8_240210_v2.txt', all_batch_x, delimiter=' ')
    
    all_batch_x = all_batch_x.reshape(-1, 64*64)
    # fn = save_path + 'radar_8_240210.txt'
    if type=='MinMaxScaler':
        save_path = save_path + 'min_max_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train_minmax = min_max_scaler.fit_transform(all_batch_x)
        dump(min_max_scaler, open(save_path+'min_max_scaler_8_240210.pkl', 'wb'))
        x_train_minmax = x_train_minmax.reshape(-1)
        fn = save_path + 'min_max_scaler_8_240210_trainx.txt'
        np.savetxt(save_path+'min_max_scaler_8_240210_trainx.txt', x_train_minmax , delimiter=' ')
        # with open(fn,'a') as file_obj:
        #     file_obj.write(str(x_train_minmax))
    if type=='StandardScaler':
        save_path = save_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # scaler = preprocessing.StandardScaler().fit(X)
        srandard_scaler = preprocessing.StandardScaler()
        x_train_srandard = srandard_scaler.fit_transform(all_batch_x)
        dump(srandard_scaler, open(save_path+'srandard_scaler_8_240210.pkl', 'wb'))
        x_train_srandard = x_train_srandard.reshape(-1)
        fn = save_path + 'srandard_scaler_8_240210_trainx.txt'
        np.savetxt(save_path+'srandard_scaler_8_240210_trainx.txt', x_train_srandard , delimiter=' ')
        # with open(fn,'a') as file_obj:
    if type == 'Normalizercaler':
        from sklearn.preprocessing import Normalizer
        x_train_normalizer = x_train_normalizer.reshape(-1)
        save_path = save_path + 'normalizer_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        normalizer_scaler = preprocessing.Normalizer().fit(all_batch_x)
        x_train_normalizer = normalizer_scaler.transform(all_batch_x)
        dump(normalizer_scaler, open(save_path+'normalizer_scaler_8_240210.pkl', 'wb'))
        fn = save_path + 'normalizer_scaler_8_240210_trainx.txt'
        np.savetxt(save_path+'normalizer_scaler_8_240210_trainx.txt', x_train_normalizer , delimiter=' ')
if __name__ == "__main__":                
    print('Initializing models')
    model = Model(args)
    gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','), dtype=np.int32)
    args.n_gpu = len(gpu_list)
    
    print("args.n_gpu=",args.n_gpu)
    # Env setting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) 
    K.set_session(sess)
    
    print("cuda = ",config.gpu_options.allow_growth ,"gpu_list=",gpu_list)
    
    
    model_parameter = {"input_shape": [512,512],
                   "output_shape": [512,512],
                   "period": 6,
                   "predict_period": 6}#,
    #               "filter": 36,
    #               "kernel_size": [1, 1]}
  

    date_date=[['2017-04-21 10:00', '2017-04-21 17:00'],
                ['2017-06-01 14:00', '2017-06-01 23:50'],
                ['2017-06-02 01:00', '2017-06-02 23:00'],
                
                ['2017-06-14 01:00', '2017-06-14 02:00'],
                ['2017-06-14 06:00', '2017-06-14 07:00'],
                
                ['2017-06-16 19:20', '2017-06-16 20:20'],
             
                ['2017-06-18 08:00', '2017-06-18 10:30'],
                
                ['2017-07-29 04:00', '2017-07-29 13:00'],
                ['2017-07-29 23:40', '2017-07-29 23:50'],

                ['2017-07-30 11:00', '2017-07-30 23:50'],
                ['2017-07-31 02:00', '2017-07-31 10:00'],
          
                ['2017-09-29 16:00', '2017-09-29 18:00'],
                ['2017-10-12 22:00', '2017-10-12 23:50'],
                ['2017-10-13 10:00', '2017-10-13 16:00'],
                ['2017-10-14 00:00', '2017-10-14 04:00'],
                ['2017-10-14 20:00', '2017-10-14 23:50'],
              

                ['2018-06-14 12:20', '2018-06-14 12:40'],
                ['2018-06-14 15:00', '2018-06-14 23:50'],
                ['2018-06-15 04:00', '2018-06-15 06:00'],
               
                ['2018-06-17 05:40', '2018-06-17 11:00'],
                ['2018-06-17 22:40', '2018-06-17 22:50'],
                ['2018-06-18 00:50', '2018-06-18 01:50'],#?
                ['2018-06-18 21:00', '2018-06-18 23:50'],#?

                ['2018-06-19 00:00', '2018-06-19 11:00'],
                ['2018-06-20 02:00', '2018-06-20 03:00'],
                ['2018-06-20 13:30', '2018-06-20 14:30'],

                ['2018-07-01 23:00', '2018-07-01 23:40'],
                ['2018-07-03 03:30', '2018-07-02 06:00'],
         
                ['2018-07-10 16:00', '2018-07-10 23:50'],
         
                ['2018-08-22 22:00', '2018-08-22 23:50'],
                ['2018-08-23 00:00', '2018-08-23 23:50'],
                ['2018-08-24 00:00', '2018-08-24 05:00'],

                ['2018-08-24 17:00', '2018-08-24 19:40'],
                ['2018-08-25 16:00', '2018-08-25 16:50'],
                ['2018-08-25 20:40', '2018-08-25 21:40'],

                ['2018-08-26 00:00', '2018-08-26 16:00'],
                ['2018-08-26 18:00', '2018-08-26 21:00'],

                ['2018-08-27 02:00', '2018-08-27 23:50'],
                ['2018-08-28 00:00', '2018-08-28 08:00'],
                ['2018-08-28 15:00', '2018-08-28 23:50'],

                ['2018-08-29 00:00', '2018-08-29 23:50'],
                ['2018-09-08 13:00', '2018-09-08 20:00'],
            
                ['2018-09-15 00:00', '2018-09-15 02:00'],
                ['2018-09-15 10:50', '2018-09-15 12:50'],



                ['2018-09-16 00:00', '2018-09-16 22:00'],#?
                ['2018-10-31 14:00', '2018-10-31 20:00'],
                ['2018-11-02 10:00', '2018-11-02 11:00'],
                
                ['2018-12-23 00:00', '2018-12-23 01:00'],

         
                ['2019-05-17 00:00', '2019-05-17 05:00'],
                ['2019-05-17 09:00', '2019-05-17 10:00'],
                ['2019-05-18 02:30', '2019-05-18 03:30'],
                ['2019-05-18 05:30', '2019-05-18 06:30'],

       
                ['2019-06-10 00:00', '2019-06-10 06:00'],
                ['2019-06-10 16:30', '2019-06-10 23:50'],

                ['2019-06-11 00:00', '2019-06-11 06:00'],
                ['2019-06-11 17:00', '2019-06-11 18:00'],
                ['2019-06-11 20:00', '2019-06-11 21:00'],

                ['2019-06-12 03:30', '2019-06-12 04:00'],
                ['2019-06-12 09:00', '2019-06-12 10:00'],

                ['2019-06-13 09:20', '2019-06-13 09:30'],
                ['2019-06-13 13:30', '2019-06-13 23:00'],


                ['2019-06-14 02:00', '2019-06-14 04:00'],
                ['2019-06-23 00:00', '2019-06-23 10:00'],
                ['2019-06-23 20:00', '2019-06-23 20:10'],
                ['2019-06-23 22:20', '2019-06-23 22:50'],


                ['2019-06-25 03:00', '2019-06-25 05:00'],
                ['2019-06-25 09:00', '2019-06-25 10:00'],

                ['2019-07-18 00:00', '2019-07-18 03:30'],
                ['2019-07-18 12:00', '2019-07-18 13:00'],
                ['2019-07-18 20:00', '2019-07-18 21:00'],


                ['2019-07-19 02:00', '2019-07-19 02:30'],
                ['2019-07-19 12:00', '2019-07-19 12:30'],
                ['2019-07-19 23:40', '2019-07-19 23:50'],

       
                ['2019-07-21 02:00', '2019-07-21 03:30'],
                ['2019-07-21 19:00', '2019-07-21 22:00'],

                ['2019-07-22 00:00', '2019-07-22 02:00'],
             
                ['2019-08-08 04:00', '2019-08-08 06:00'],
                ['2019-08-08 10:00', '2019-08-08 14:00'],
                ['2019-08-08 23:40', '2019-08-08 23:50'],
             
                ['2019-08-09 15:00', '2019-08-09 17:00'],
             
                ['2019-08-13 00:00', '2019-08-13 02:00'],

                ['2019-08-14 03:30', '2019-08-14 03:40'],
                ['2019-08-14 20:00', '2019-08-14 23:00'],
                ['2019-08-14 23:40', '2019-08-14 23:50'],


                ['2019-08-15 04:00', '2019-08-15 04:10'],
                ['2019-08-15 18:00', '2019-08-15 20:00'],
                ['2019-08-15 23:40', '2019-08-15 23:50'],

                ['2019-08-16 04:00', '2019-08-16 05:00'],
                ['2019-08-16 22:00', '2019-08-16 22:30'],

             
        
                ['2019-08-18 04:00', '2019-08-18 15:00'],
                ['2019-08-18 23:00', '2019-08-18 23:50'],

                ['2019-08-19 02:00', '2019-08-19 07:00'],
                ['2019-08-20 08:00', '2019-08-20 08:30'],
                ['2019-08-24 01:00', '2019-08-24 16:00'],
             
                ['2019-09-26 22:00', '2019-09-26 23:50'],
                ['2019-09-27 01:00', '2019-09-27 08:00'],
                ['2019-09-28 06:30', '2019-09-28 07:00'],
               
                ['2019-09-29 18:00', '2019-09-29 23:50'],
                ['2019-09-30 02:00', '2019-09-30 08:00'],
                ['2019-09-30 16:00', '2019-09-30 17:00'],
              
                ['2019-12-29 01:00', '2019-12-29 08:00'],
                ['2019-12-29 15:10', '2019-12-29 15:20'],
                
                ['2019-12-30 04:30', '2019-12-29 06:00']]


    test_date=[['2017-10-14 00:00', '2017-10-14 00:10'],
             ['2018-08-24 00:00','2018-08-24 23:59'],
             ['2019-05-16 00:00','2019-05-16 23:59'],
             ['2019-08-14 00:00','2019-08-14 23:59'],
             ['2020-05-27 00:00','2020-05-27 23:59']]
    places=['Sun_Moon_Lake']

    # radar_echo_storage_path= None
    radar_echo_storage_path= 'E:/yu_ting/try/NWP/'#'NWP/'
    load_radar_echo_df_path='PredRNN_L4_H64_2017to2019_pick_day_v4/2017to2018_pick_day.pkl'

    save_path ='PredRNN_L4_H64_2017to2019_pick_day_v4/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    radar = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                    load_radar_echo_df_path=load_radar_echo_df_path,
                    input_shape=model_parameter['input_shape'],
                    output_shape=model_parameter['output_shape'],
                    period=model_parameter['period'],
                    predict_period=model_parameter['predict_period'],
                    places=places,
                    random=False,
                    date_range=date_date,
                    test_date=test_date,
                    save_np_radar=save_path )
    # if not load_radar_echo_df_path:
        # radar.exportRadarEchoFileList()
    # if not os.path.isdir(save_path):
        # os.makedirs(save_path)
        # radar.saveRadarEchoDataFrame(path=save_path ,load_name_pkl='T12toT6_2017_2018_2019_pickday_512x512_')   
    
    # sys.exit()
#    save_path = 'save_3m_itr10_0916/'
    # save_path ='save_2y3m_p6_0923_batchsize32_newmodel/'
    
    # save_path ='save_8240110_same_traintest_over_fitting_model_StandardScaler/'

    # sys.exit()
    # save_path ='save_8230110_novaltest_over_fitting_model_itr2000_changing_rate0.0005_model_LayerNormpy_test_1900/'
#    'save_2y3m_0918/'
#    model_name = 'mode_haveLayerNorm_3m_itr3.pkl'
    # model_name = 'mode_haveLayerNorm_2y3m_p6_new_model'
    # model_name = 'model_LayerNormpy_824_Sun_Moon_Lake_model'#'mode_haveLayerNorm_2y3m_p4_new_modelitr7.pkl'

    # model_name = 'model_LayerNormpy_8240110_novaltest_Sun_Moon_Lake_model_itr500.pkl'
    # model_name = 'model_824_Sun_Moon_Lake_itr231.pkl'#'mode_haveLayerNorm_2y3m_p4_new_modelitr7.pkl'
    # model_name = 'model_823_Sun_Moon_Lake_itr1869.pkl'
#    save_path = 'save_2y3m_0918/'
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Normalizer
    # Normalizer().fit(X)
    train_radar_xy_shuffle = None
    val_radar_xy_shuffle = None
    pretrained_model = None#'model_itr2_test_cost21.75484814650153_ssim0.678033175804233.pkl'#'p1_model_itr26_test_cost6.151229987320058_ssim0.8462856699373386.pkl'#None
    model_name ='model' #'p1_model_itr5_test_cost9.519979798220062_ssim0.8119851203114509.pkl'

    if args.is_training:
    #    scaler = StandardScaler()
        # muti_sample(save_path,type ='StandardScaler')#!
        # sys.exit()
        # train_radar_xy_shuffle, val_radar_xy_shuffle = load_sample(save_path)#!
        train_sample_wrapper(model, save_path,pretrained_model , model_name, train_radar_xy_shuffle, val_radar_xy_shuffle)#!
        
        # train_sample_wrapper(model, save_path, model_name, train_radar_xy_shuffle, val_radar_xy_shuffle)#!
        # sys.exit()
        # preprocess_sample(save_path,type ='StandardScaler')
        # preprocess_fun(save_path,type ='StandardScaler')# 'StandardScaler')#!
        # model_name = 'model_8240210_Sun_Moon_Lake_model_itr28914_test_cost0.00023020233493298292.pkl'
        # train_wrapper(model, save_path, model_name )#, train_radar_xy_shuffle, val_radar_xy_shuffle)#!
    #    model_name = 'model_LayerNormpy_8230010_novaltest_Sun_Moon_Lake_model_test_itr1900.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr15000_test_cost0.560447613398234.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr10000.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr15000_test_cost0.026278110841910046.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr20000_test_cost0.006796003629763921.pkl'
    #    model_name ='model_8240110_Sun_Moon_Lake_model_itr30000_test_cost0.0007234304212033749.pkl'
        # model_name ='model_8240210_0310_Sun_Moon_Lake_model_itr34597_test_cost0.00037630179819340504.pkl'
        # model_name = 'model_8240210_0310_Sun_Moon_Lake_model_itr185_test_cost3.1776345775001946.pkl'
        # model_name = 'p1_model_itr19_test_cost7.495534596535786_ssim0.7783282523016555.pkl'
        # model_name = 'pertrain_model_201807to08_add_201706_itr7_test_cost6.9234249002449655_ssim0.8543377402331324.pkl'
        # test_wrapper(model, save_path, model_name,itr=26,load_model=True)
      
        # test_show(model, save_path, model_name,itr=5)
        # for i in range(1,21):
        #     model_name = 'p1_model_itr{}.pkl'.format(i)
        #     test_wrapper(model, save_path, model_name,itr=i,load_model=True)
    
    # else:
    #   val(model, save_path,model_pkl = model_name, itr=10)
        # tw(model, save_path, model_name, itr=1)


        # for period in range(model_parameter['predict_period']):
        #     #    print('pred_y[:, period] = ', pred_y[:, period])
        #     #    print('test_y[:, period] = ', test_y[:, period])
        #     csi_eva = Verification(pred=pred_y[:, period].reshape(-1, 1), target=test_y[:, period].reshape(-1, 1), threshold=60, datetime='')
        #     csi.append(np.nanmean(csi_eva.csi, axis=1))
        
        # csi = np.array(csi)
        # np.savetxt(save_path+'T202005270000csi_reshape1.csv', csi, delimiter = ',')
        # np.savetxt(save_path+'T202005270000csi.csv', csi.reshape(6,60), delimiter = ' ')

        # ## Draw thesholds CSI
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        # ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        # plt.xlim(0, 60)
        # plt.ylim(-0.05, 1.0)
        # plt.xlabel('Threshold')
        # plt.ylabel('CSI')
        # plt.title('20200527day\nThresholds CSI')
        # plt.grid(True)

        # all_csi = []
        # for period in range(model_parameter['predict_period']):
        # plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), 'o--', label='{} min'.format((period+1)*10))

        # plt.legend(loc='upper right')

        # fig.savefig(fname=save_path+'Thresholds_CSI.png', format='png')
        # plt.clf()


        # ## Draw thesholds AVG CSI
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        # ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        # plt.xlim(0, 60)
        # plt.ylim(-0.05, 1.0)
        # plt.xlabel('Threshold')
        # plt.ylabel('CSI')
        # plt.title('20200527\nThresholds CSI')
        # plt.grid(True)

        # all_csi = []
        # plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), 'o--', label='AVG CSI')
        
        # plt.legend(loc='upper right')

        # fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
        # plt.clf()

        # #csie = time.clock()
        # #
        # #alle = time.clock()
        # #
        # #print("load NWP time = ", loadNe - loadNs)
        # #print("load CREF time = ", loadCe - loadCs)
        # #print("All time = ", alle - alls)
        # ## Draw peiod ALL CSI 
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        # ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        # plt.xlim(0, model_parameter['predict_period']+1)
        # plt.ylim(-0.05, 1.0)
        # plt.xlabel('Time/10min')
        # plt.ylabel('CSI')
        # my_x_ticks = np.arange(0, model_parameter['predict_period']+1, 1)
        # plt.xticks(my_x_ticks)
        # plt.title('Threshold 5-55 dBZ')
        # plt.grid(True)
        # i = 0
        # for threshold in range(5, 56, 5):
        # plt.plot(np.arange(len(csi)+1), [np.nan] + list(csi[:, threshold-1]), 'o--', label='{} dBZ'.format(threshold), color=Color[i])
        # i = i + 1
        # #plt.legend(loc='lower right')

        # plt.clf()

        # fig.savefig(fname=save_path+'Period_CSI_ALL2.png', format='png')


        # rmse=np.sqrt(((pred_list - test_y) ** 2).mean())
        # fn = save_path + 'p20_20200527_rmse.txt'
        # with open(fn,'a') as file_obj:
        #     file_obj.write('rmse=' + str(rmse)+'\n')

        #     # file_obj.write('mse=' + str(mse)+'\n')