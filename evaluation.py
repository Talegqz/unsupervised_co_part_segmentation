import os
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from tqdm import trange
from sklearn import linear_model
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from time import gmtime, strftime
import numpy as np
import cv2
import imageio
from skimage import io, img_as_float32
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import model as net
import cv2

class FramesDataset(data.Dataset):
    def __init__(self, root_dir, is_train=True, root_dir_masks=None):
        super(FramesDataset, self).__init__()
        self.is_train = is_train
        self.root_dir_masks = root_dir_masks

        train_images, test_images = os.listdir(os.path.join(root_dir, 'train')), os.listdir(os.path.join(root_dir, 'test'))

        if self.is_train:
            self.images = train_images
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.images = test_images
            self.root_dir = os.path.join(root_dir, 'test')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        path = os.path.join(self.root_dir, name)

        video = img_as_float32(io.imread(path))

        out = {}
        image = np.array(video, dtype='float32')

        out['img'] = image.transpose((2, 0, 1)).astype('float32')
        out['name'] = name

        if self.root_dir_masks is not None:
            path_mask = os.path.join(self.root_dir_masks, name)
            mask = img_as_float32(io.imread(path_mask))
            out['mask'] = np.expand_dims(mask, axis=0)

        return out

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1)) / x_max * 2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T / y_max * 2 - 1.0

    x_map = torch.from_numpy(x_map.astype(np.float32))
    y_map = torch.from_numpy(y_map.astype(np.float32))

    return x_map, y_map

def get_center(part_map):
    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    return x_center, y_center


def sort_column(df):
    df = df.sort_values('file_name')
    return df


def regress_keypoints(df_kp):
    # train dataframes
    train_gnd = sort_column(df_kp['train_gnd'])
    train_pred = sort_column(df_kp['train_pred'])

    # test dataframes
    test_gnd = sort_column(df_kp['test_gnd'])
    test_pred = sort_column(df_kp['test_pred'])

    # convert dataframe to numpy
    train_gnd = np.stack(train_gnd['value'].values)
    train_pred = np.stack(train_pred['value'].values)

    test_gnd = np.stack(test_gnd['value'].values)
    test_pred = np.stack(test_pred['value'].values)

    scores = []
    num_gnd_kp = train_gnd.shape[1]
    for i in range(num_gnd_kp):
        for j in range(2):
            # print('Fitting linear model for...{},{}'.format(i, j))
            index = train_gnd[:, i, j] != -1
            features = train_pred[index]
            features = features.reshape(features.shape[0], -1)
            label = train_gnd[index, i, j]
            reg = linear_model.LinearRegression()
            reg.fit(features, label)

            index_test = test_gnd[:, i, j] != -1
            features = test_pred[index_test]
            features = features.reshape(features.shape[0], -1)
            label = test_gnd[index_test, i, j]
            #score = reg.score(features, label) # using sklearn's score
            score = np.mean(np.abs(reg.predict(features) - label))
            scores.append(score)
    print(np.sum(scores))
    return(np.sum(scores))

def evaluate_landmark(segmentation_module, dataset,save_name,dataset_name_pkl,data_path_base):
    trainloader = data.DataLoader(dataset['train'], batch_size=1, shuffle=False, drop_last=False)
    testloader = data.DataLoader(dataset['test'], batch_size=1, shuffle=False, drop_last=False)

    # put in eval mode
    segmentation_module.train()

    size = (256,256)
    Train_finshed = False
    Test_finshed = False
    # iterate over train images to obtain predicted keypoints for train set
    print('Computing keypoints on train set. Please wait...')
    # obtain keypoint for train images
    out_df = {'file_name': [], 'value': []}
    lms_pred = []
    train_iter = iter(trainloader)
    with torch.no_grad():
        for _ in trange(len(trainloader.dataset)):
            # if _>10:
            #     break
            if Train_finshed:
                break
            batch = train_iter.next()
            image = batch['img']
            name = batch['name'][0]
            image = torch.nn.functional.interpolate(image,scale_factor=0.5)
            # get the model output
            output = segmentation_module(image.cuda(),image.cuda())['pred_image_t_weight'][0,:,0,:,:]
            w,h=128,128
            mask_image = torch.zeros((w,h))
            mask_image_weight = torch.zeros((w,h))
            for weight_index,this_weight in enumerate(output):
                this_weight = this_weight.cpu()
                mask_image[this_weight>mask_image_weight] = weight_index
                mask_image_weight[this_weight>mask_image_weight] = this_weight[this_weight>mask_image_weight]
            
            output2 = torch.zeros(output.shape)
            for weight_index,this_weight in enumerate(output):
                output2[weight_index][mask_image==weight_index] = 1
                this_image = (output2[weight_index].numpy()*255).astype(np.uint8)
                # cv2.imwrite('temp_result/%d_%d.png'%(_,weight_index),this_image)
            output = output2.unsqueeze(0)
            output = F.interpolate(output.cpu(), size=size, mode='bilinear')
            centers = []
            for j in range(0, output.shape[1]-1):  # ignore the background
                part_map = output[0, j, ...] + 1e-8
                k = part_map.sum()
                part_map_pdf = part_map / k
                x_c, y_c = get_center(part_map_pdf)
                x_c = (x_c + 1.) / 2 * size[0]
                y_c = (y_c + 1.) / 2 * size[0]
                center = torch.stack((x_c, y_c), dim=0).unsqueeze(0)  # compute center of the part map
                centers.append(center)
                # print(center)
            centers = torch.cat(centers, dim=0)
            lms_pred.append(centers.unsqueeze(0))
            out_df['value'].append(centers.numpy())
            out_df['file_name'].append(name)

        # save the landmarks in a pandas dataframe
        kp_train_df = pd.DataFrame(out_df)
        kp_train_df.to_pickle('%s/'%data_path_base + dataset_name_pkl + '_train_pred%s.pkl'%save_name)

    # iterate over the test images to obtain predicted keypoints for test set
    print('Computing keypoints on test set. Please wait...')
    # obtain keypoint for test images
    out_df = {'file_name': [], 'value': []}
    lms_pred = []
    test_iter = iter(testloader)
    with torch.no_grad():
        for _ in trange(len(testloader.dataset)):
            if Test_finshed:
                break
            # if _>10:
            #     break
            batch = test_iter.next()
        
            image = batch['img']
            name = batch['name'][0]

            image = torch.nn.functional.interpolate(image,scale_factor=0.5)

            output = segmentation_module(image.cuda(),image.cuda())['pred_image_t_weight'][0,:,0,:,:]
            w,h=128,128
            mask_image = torch.zeros((w,h))
            mask_image_weight = torch.zeros((w,h))
            for weight_index,this_weight in enumerate(output):
                this_weight = this_weight.cpu()
                mask_image[this_weight>mask_image_weight] = weight_index
                mask_image_weight[this_weight>mask_image_weight] = this_weight[this_weight>mask_image_weight]
            
            output2 = torch.zeros(output.shape)
            for weight_index,this_weight in enumerate(output):
                output2[weight_index][mask_image==weight_index] = 1
                this_image = (output2[weight_index].numpy()*255).astype(np.uint8)
                # cv2.imwrite('temp_result/%d_%d.png'%(_,weight_index),this_image)
            output = output2.unsqueeze(0)
            output = F.interpolate(output.cpu(), size=size, mode='bilinear')
            centers = []
            for j in range(0, output.shape[1]-1):  # ignore the background
                part_map = output[0, j, ...] + 1e-8
                k = part_map.sum()
                part_map_pdf = part_map / k
                x_c, y_c = get_center(part_map_pdf)
                x_c = (x_c + 1.) / 2 * size[0]
                y_c = (y_c + 1.) / 2 * size[0]
                center = torch.stack((x_c, y_c), dim=0).unsqueeze(0)  # compute center of the part map
                centers.append(center)
                
            centers = torch.cat(centers, dim=0)
            lms_pred.append(centers.unsqueeze(0))

            out_df['value'].append(centers.numpy())
            out_df['file_name'].append(name)


        kp_train_df = pd.DataFrame(out_df)
        kp_train_df.to_pickle('%s/'%data_path_base + dataset_name_pkl + '_test_pred%s.pkl'%save_name)


def start_test_val(save_name,dataset_name_pkl,data_path): 
     # regress from predicted keypoints to ground truth landmarks
    df_kp = {}
    df_kp['train_gnd'] = pd.read_pickle('%s/'%data_path + dataset_name_pkl + '_train_gt.pkl')
    df_kp['train_pred'] = pd.read_pickle('%s/'%data_path+ dataset_name_pkl + '_train_pred%s.pkl'%save_name)
    df_kp['test_gnd'] = pd.read_pickle('%s/'%data_path + dataset_name_pkl + '_test_gt.pkl')
    df_kp['test_pred'] = pd.read_pickle('%s/'%data_path + dataset_name_pkl + '_test_pred%s.pkl'%save_name)
    return(regress_keypoints(df_kp))


def init_model(net,load_path,part_numb):
    model = net.Part_3D_Disnet(part_numb)
    model.train()
    # model.eval()
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(load_path))
    return model

def evaluate_iou(segmentation_module, dataset):
   
    testloader = data.DataLoader(dataset['test'], batch_size=1, shuffle=False, drop_last=False)

    # put in eval mode
    segmentation_module.train()
    size = (256,256)

    lms_pred = []
    test_iter = iter(testloader)
    all_iou_list = []
    with torch.no_grad():
        for _ in trange(len(testloader.dataset)):
     
            batch = test_iter.next()
        
            image = batch['img']
            name = batch['name'][0]
            mask = batch['mask'][0][0]
           
            image = torch.nn.functional.interpolate(image,size=(128,128))
            output = segmentation_module(image.cuda(),image.cuda())['pred_image_t_weight'][0,:,0,:,:]
            w,h=128,128
            mask_image = torch.zeros((w,h))
            mask_image_weight = torch.zeros((w,h))
            for weight_index,this_weight in enumerate(output):
                if weight_index!=len(output)-1:

                    this_weight = this_weight.cpu()
                    mask_image[this_weight>mask_image_weight] = weight_index
                    mask_image_weight[this_weight>mask_image_weight] = this_weight[this_weight>mask_image_weight]
                else:
                    this_weight = this_weight.cpu()
                    temp_mask  = torch.zeros(this_weight.shape)+0.5
                    mask_image[(this_weight>mask_image_weight)&(this_weight>0.8)] = weight_index
                    mask_image_weight[this_weight>mask_image_weight] = this_weight[this_weight>mask_image_weight]

            output2 = torch.zeros((1,128,128))
            output2[0][mask_image==weight_index] = -1
            output2 = output2+1
            output = output2.unsqueeze(0)
            our_result_mask = F.interpolate(output.cpu(), size=size, mode='bilinear')[0][0].cpu().detach().numpy()
            mask[mask>0] = 1
            ground_truth = mask.cpu().detach().numpy()

            intersection = our_result_mask*ground_truth

            intersection_count = intersection.sum()
            all_ground_count = ground_truth.sum()
            all_our_count    =  our_result_mask.sum()
            final_iou = intersection_count/(all_our_count+all_ground_count-intersection_count)

            all_iou_list.append(final_iou)

    print(sum(all_iou_list)/len(all_iou_list))
    return sum(all_iou_list)/len(all_iou_list)



def evaluation(dataset_name,model_path,data_path_base,part_numb=11,save_name='_'):
    dataset = {}
    data_path = '%s/%s-256'%(data_path_base,dataset_name)
    root_dir_masks = '%s/mask/%s-test-masks'%(data_path_base,dataset_name)
    dataset['train'] = FramesDataset(root_dir=data_path, is_train=True)
    dataset['test'] = FramesDataset(root_dir=data_path, is_train=False,root_dir_masks=root_dir_masks)
    
    model = init_model(net,model_path,part_numb)
    result_v = evaluate_iou(model,dataset)
    print('%s IoU result '%dataset_name,result_v)
    evaluate_landmark(model, dataset, save_name, dataset_name,data_path_base)
    result_p = start_test_val(save_name,dataset_name,data_path_base)


    print('%s Landmark result '%dataset_name,result_p)


    pass
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        default='',
                        type=str,
                        required=True)
    parser.add_argument('--data_path',
                        default='evaluation_data/taichi_and_vox',
                        type=str)
    parser.add_argument('--model_path',
                        default='save_model/',
                        type=str)


    arg = parser.parse_args()
    
    evaluation(arg.dataset_name,arg.model_path,arg.data_path)
    # python evaluation.py --dataset_name vox --model_path save_model/vox --data_path evaluation_data/taichi_and_vox
    
   


    


