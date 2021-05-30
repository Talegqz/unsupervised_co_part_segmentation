import model as net
from argparse import ArgumentParser
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
def read_video(name, frame_shape=(128,128)):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = []
        for idx in range(num_frames):
            this_image= cv2.resize(
                    cv2.cvtColor(
                        cv2.imread(os.path.join(name, frames[idx]))
                        ,cv2.COLOR_BGR2RGB
                        )
                    ,(frame_shape[0:2])
                    ) 
            video_array.append(
                this_image
                )
        video_array = np.array(video_array)/255

    return video_array

def out_tensor_to_image(out_tensor):
    out_image = out_tensor.detach().cpu().numpy()
    out_image[out_image>1] = 1
    out_image[out_image<0] = 0
    out_image = out_image*255
    out_image = out_image.transpose(1,2,0).astype(np.uint8)

    return out_image

def init_model(model_path,part_numb):
    model = net.Part_3D_Disnet(part_numb).cuda()
    model.train()
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    
    return model

def start_run(opt):
    colormap = plt.get_cmap('gist_rainbow')
    model = init_model(opt.checkpoint_path,opt.part_numb)
    all_data = read_video(opt.driving_path)
    source_image_base = cv2.imread(opt.source_image)
    source_image_base = cv2.resize(source_image_base,(128,128))
    source_image = source_image_base[np.newaxis,:,:,:]
    source_image = source_image.transpose(0,3,1,2)/255
    source_image = torch.from_numpy(source_image).float()
    source_image = torch.nn.functional.interpolate(source_image,(128,128)).cuda()
    if not os.path.exists(opt.out_dir+'/images'):
        os.makedirs(opt.out_dir+'/images')
    video_lenth = len(all_data)
    # video_writer = cv2.VideoWriter(opt.out_dir+'/video.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 15, (128*3, 128), True)
    final_image_list = []
    for index_frame in range(video_lenth):
        driving_image_base = all_data[index_frame:index_frame+1]
        driving_image = driving_image_base.transpose(0,3,1,2)
        driving_image = torch.from_numpy(driving_image)
        driving_image = driving_image.float()
        driving_image = torch.nn.functional.interpolate(driving_image,(128,128)).cuda()
        out = model(source_image,driving_image)
        pred_image = out['pred_image_t'][0]
        pred_image = out_tensor_to_image(pred_image)
        driving_image = cv2.resize(driving_image_base[0],(128,128))
        driving_image =(driving_image*255).astype(np.uint8)
        
        final_image = np.hstack([source_image_base,pred_image,driving_image])
        print(final_image.shape)
        final_image_list.append(final_image.copy())
        final_image =  cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(opt.out_dir+'/images/%04d.png'%index_frame,final_image)
    imageio.mimsave(opt.out_dir+'/video.gif',final_image_list,fps=10)
       

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script")
 
    parser.add_argument("--driving_path", required=True, help="path to driving image")
    parser.add_argument("--source_image", required=True, help="path to source image")
    parser.add_argument("--checkpoint_path", required=True,help="path to checkpoint to restore")
    parser.add_argument("--out_dir", required=True,help="path to save_result")
    parser.add_argument("--part_numb", default=11,type=int,help="model_part_numb")
    opt = parser.parse_args()

    start_run(opt)
