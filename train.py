import sys
sys.path.append('..')

import numpy as np
import torch
import torch.utils.data as Data

import util
import json
from argparse import Namespace

import progressbar as probar
from collections import OrderedDict
import random
import os
from tensorboardX import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import GetDataSet
import cv2
import time
import model as net
# class Main():
#     def __init__(self):
#         pass
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3"
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# temp_dir = 'ztemp_p'
# util.mkdir(temp_dir)
color_space = util.colorspace()
def visual_part(part_tensor):
    
    part_numbs,d,w,h = part_tensor.shape
    image_list_tensor_list = []
    
    for part_index in range(part_numbs):
        # print(part_index)
        this_part =  part_tensor[part_index].detach().cpu().numpy()
        this_part =  part_tensor[part_index].detach().cpu().numpy()
        
        
        # print(this_part.max())
        # print(this_part.min())
        this_part[this_part<0.2]=0
        this_scatter =  np.where(this_part)
        ax.scatter(this_scatter[0], this_scatter[1], this_scatter[2])
        plt.savefig(temp_dir+'/test_scaltter%d.png'%part_index)
        plt.cla()

        this_image = cv2.imread(temp_dir+'/test_scaltter%d.png'%part_index)
        this_image = (this_image/255.).transpose(2,0,1)
        this_image = torch.from_numpy(this_image)
        image_list_tensor_list.append(this_image)                                                                                                             
    return image_list_tensor_list
 

def init_model():
    model = net.Part_3D_Disnet(part_numb=arg.part_numb)
    model.train()
    model.cuda()
    model = torch.nn.DataParallel(model)


    if arg.load_path is not None:
        print('we are loading！')
        model.load_state_dict(torch.load(arg.load_path))


    return model
def out_tensor_to_image(out_tensor):
    out_image = out_tensor.detach().cpu().numpy()
    out_image[out_image>1] = 1
    out_image[out_image<0] = 0
    out_image = out_image*255
    out_image = out_image.transpose(1,2,0).astype(np.uint8)

    return out_image
def image_numpy_tensor(x):
    x = np.array(x)
    x = x/255.
    x = torch.from_numpy(x)
    x = x.float()
    return x
def start_train():
    
    model = init_model()

    this_image_name = save_name.split('/')[-1]
    print(this_image_name)
    logger.info(str(arg)) 

    optimizer = torch.optim.Adam([{
        'params':
        filter(lambda p: p.requires_grad, model.parameters()),
        'lr':
        arg.lr
    }])

    data_loader = GetDataSet.get_dataset(arg.data_path,arg.batch_size,repeat_numb=arg.repeat_numb,arg=arg)
    
    data_size = len(data_loader)


    total_steps = arg.total_steps
    for epoch in range(arg.epoch_start, arg.epoch_bound):
        epoch_iter= 0
        widgets = [
            'Progress: ',
            probar.Percentage(), ' ',
            probar.Bar('#'), ' ',
            probar.Timer(), ' ',
            probar.ETA(), ' ',
            probar.FileTransferSpeed()
        ]
        # if ""
        if "rotation_param" in arg.augmentation_params and epoch>=arg.need_more_rotation:
            arg.augmentation_params["rotation_param"]["degrees"] += arg.increase_degrees
            if arg.augmentation_params["rotation_param"]["degrees"]>=180:
                arg.augmentation_params["rotation_param"]["degrees"]  = 180
            print(arg.augmentation_params["rotation_param"]["degrees"],'rotation_param degrees')
        if epoch>=arg.need_less_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * arg.lr_delay
            arg.need_less_lr+=10     
        data_loader = GetDataSet.get_dataset(arg.data_path,arg.batch_size,repeat_numb=arg.repeat_numb,arg=arg)
        if epoch>=arg.need_more_shift:
            arg.weight_shift = arg.weight_shift*10
            arg.weight_matrix = arg.weight_matrix*10
            arg.need_more_shift = 1000
        print(arg)
        print(arg.save_name)

        def vis(generated_dict,loss_dict):
            generated_dict['pred_image_s'] = torch.clamp(generated_dict['pred_image_s'], min=0, max=1).detach().cpu()
            generated_dict['pred_image_t'] = torch.clamp(generated_dict['pred_image_t'], min=0, max=1).detach().cpu()


            generated_dict['pred_latent_s'] = torch.clamp(generated_dict['pred_latent_s'], min=0, max=1).detach().cpu()
            # image_kp,latent_kp = keypoint_wirte_on_image(source_image[0],generated_dict['s_now'][0],generated_dict['s_relative'][0])
            this_image_list = [
                            source_image[0],
                            generated_dict['pred_image_s'][0],
                            generated_dict['image_target'][0],
                            generated_dict['pred_image_t'][0],
                            generated_dict['pred_latent_s'][0]
                            ]
            final_im = util.merge_images_tensor(this_image_list) 
            writer.add_image('final_im',final_im,total_steps,dataformats='CHW')

            losses_dict = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_dict.items()}
            losses_dict['a_loss_all'] = float(loss_values.detach().data.cpu().numpy())
            logger.info("epoch %.3f" %
                        (epoch + float(epoch_iter / data_size)))
            logger.info("loss %.3f" %
                        (float(loss_values.detach().data.cpu().numpy())))
            print(arg.save_name)
            for loss_name in losses_dict:
                writer.add_scalar(loss_name,losses_dict[loss_name],total_steps)
        
            this_image_list = []
            part_numb = generated_dict['pred_image_t_part'].shape[1]
            for j in range(part_numb):
                this_image_list.append(torch.clamp(generated_dict['pred_image_t_part'][0][j],min=0,max=1))

            final_im = util.merge_images_tensor(this_image_list) 
            writer.add_image('pred_image_t_part',final_im,total_steps,dataformats='CHW')
            this_image_list = []
            each_part_center = net.batch_get_centers(generated_dict['pred_image_t_weight'][:,:,0,:,:])[0].cpu().detach().numpy()
            for weight_index in range(part_numb):
                k_p = generated_dict['t_shift'].cpu().detach().numpy()[0][weight_index]
                this_part = generated_dict['pred_image_t_weight'][0][weight_index]*0.8+generated_dict['image_target'][0]*0.2
                out_t_image_real = out_tensor_to_image(this_part)
                out_t_image_real = cv2.cvtColor(out_t_image_real,cv2.COLOR_BGR2RGB)
                out_t_image_real = cv2.cvtColor(out_t_image_real,cv2.COLOR_BGR2RGB)

                x_p = ((k_p[0]*(1)+1)/2*128).astype(np.int32)
                y_p = ((k_p[1]*(1)+1)/2*128).astype(np.int32)

                this_color = (color_space[weight_index+1]*255).astype(np.int32)

                cv2.circle(out_t_image_real, (x_p,y_p), 4, (int(this_color[0]),int(this_color[1]),int(this_color[2])), -1)
                
                k_p = each_part_center[weight_index]
                x_p = ((k_p[0]+1)/2*128).astype(np.int32)
                y_p = ((k_p[1]+1)/2*128).astype(np.int32)
                this_color = (color_space[weight_index+1]*255).astype(np.int32)
                cv2.circle(out_t_image_real, (x_p,y_p), 4, (int(this_color[0]),int(this_color[1]),int(this_color[2])), -1)
                
                out_t_image_real = out_t_image_real.transpose(2,0,1)
                out_t_image_real = image_numpy_tensor(out_t_image_real)

                this_image_list.append(out_t_image_real)
            
            final_im = util.merge_images_tensor(this_image_list) 
            writer.add_image('pred_image_t_weight',final_im,total_steps,dataformats='CHW')

            this_image_list = []
            for j in range(part_numb):
                this_image_list.append(generated_dict['pred_image_s_part'][0][j])
            final_im = util.merge_images_tensor(this_image_list) 
            writer.add_image('pred_image_s_part',final_im,total_steps,dataformats='CHW')

            this_image_list = []
            for j in range(part_numb):
                this_image_list.append(generated_dict['pred_image_s_weight'][0][j])
            final_im = util.merge_images_tensor(this_image_list) 
            writer.add_image('pred_image_s_weight',final_im,total_steps,dataformats='CHW')
       
            del loss_dict
            del this_image_list
   

        for i, data in enumerate(data_loader):
            epoch_iter += arg.batch_size
            total_steps += arg.batch_size
            if i == 0:
                print('start bar')
                pbar = probar.ProgressBar(
                    widgets=widgets,
                    maxval=data_size,
                ).start()
            pbar.update(i)

            source_image = data['source'].cuda()
            inter_image   = data['driving'].cuda()

            source_image = torch.nn.functional.interpolate(source_image,(128,128))
            inter_image = torch.nn.functional.interpolate(inter_image,(128,128))

            generated_dict= model(source_image,inter_image)
    
            loss_dict = model.module.calculate_loss(generated_dict,arg=arg)


            loss_values = 0
            for  val in loss_dict:
                loss_values+=loss_dict[val].mean()
            loss_values.backward()
            optimizer.step()
            optimizer.zero_grad()
            

            if total_steps % arg.dis_step == 0:
                vis(generated_dict,loss_dict)

                
            del generated_dict
            del loss_values
            del loss_dict



        if epoch % arg.save_freq == 0:
            torch.save(model.state_dict(),
                       save_name +'/'+ str(epoch))
   




import argparse


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-b',
                        '--batch_size',
                        default=2,
                        type=int,
                        metavar='Batch',
                        help='batch_size')
    parser.add_argument('--epoch_bound',
                        default=6000000,
                        type=int,
                        metavar='epoch_bound',
                        help='epoch_bound')
    parser.add_argument('--epoch_start',
                        default=0,
                        type=int,
                        metavar='epoch_start',
                        help='epoch_start')
    parser.add_argument('--need_more_rotation',
                        default=1000,
                        type=int)
    parser.add_argument('--increase_degrees',
                        default=10,
                        type=int)
    parser.add_argument('-max_size',
                        default=1e10000,
                        type=int,
                        metavar='Batch',
                        help='frame_nums')
    parser.add_argument('--lr',
                        default=5e-5,
                        type=float,
                        metavar='Batch',
                        help='frame_nums')
    parser.add_argument('--need_less_lr',
                        default=1000,
                        type=int)
    parser.add_argument('--lr_delay',
                        default=0.98,
                        type=float,
                        metavar='Batch',
                        help='frame_nums')
    parser.add_argument('--weight_bone',
                        default=0,
                        type=float,
                        metavar='Batch',
                        help='frame_nums')
    parser.add_argument('--arg_file',
                        required=True,
                        help='arg_file_path')
    parser.add_argument('-dis',
                        '--dis_step',
                        default=2400,
                        type=int,
                        metavar='display_freq',
                        help='display_freq')
    parser.add_argument(
                        '--repeat_numb',
                        default=5,
                        type=int,
                        metavar='display_freq',
                        help='display_freq')
    parser.add_argument('-now_step',
                        '--total_steps',
                        default=0,
                        type=int,
                        metavar='total_steps',
                        help='total_steps') 
    parser.add_argument('--save_name',
                        default='save_model/V13',
                        type=str,
                        metavar='DIR',
                        help='path to where the model saved')
    parser.add_argument('-aug',
                        default=True,
                        type=bool,
                        metavar='PATH',
                        help='path to where coco images sto red')
    parser.add_argument('--load_path',
                        default=None,
                        type=str,
                        metavar='DIR',
                        help='path to load save_path')
    parser.add_argument('-dtrain',
                        '--data_path',
                        default='/mnt/gqzdataset/taichi',
                        type=str,
                        metavar='DIR',
                        help='path to load  data')
    parser.add_argument('-dtest',
                        '--data_path_test',
                        default='../data/V0/test_raw',
                        type=str,
                        metavar='DIR',
                        help='path to load  data')

    parser.add_argument('-s_len',
                        '--sequence_len',
                        default=1,
                        type=int,
                        metavar='Batch',
                        help='sequence_len')
    parser.add_argument('--save_freq',
                        default=1,
                        type=int,
                        metavar='ss',
                        help='sss')
    parser.add_argument('--model_name',
                        default='base',
                        type=str,
                        metavar='ss',
                        help='model_name')
    parser.add_argument('--augmentation_params',
                        default=None,
                        metavar='ss',
                        help='model_name')
    arg = parser.parse_args()

    return arg


arg = get_parse()
arg_dict = vars(arg)
if arg.arg_file is not None:
    with open(arg.arg_file, 'r') as f:
        arg_str = f.read()
        file_args = json.loads(arg_str)
        arg_dict.update(file_args)
        arg = argparse.Namespace(**arg_dict)
save_name = arg.save_name
util.mkdir(save_name)
logger = util.get_log(save_name + '/log.txt')
writer = SummaryWriter(save_name)

if __name__ == "__main__":

    print(arg)
    print(arg.save_name)
    start_train()
    pass


