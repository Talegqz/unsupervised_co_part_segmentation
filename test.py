import model as net
from argparse import ArgumentParser
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from imageio import mimread
from skimage.color import gray2rgb
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
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array=[]
        for v in video:
            v = cv2.resize(v
                    ,(frame_shape[0:2])
                    ) 
            video_array.append(v)
        video_array = np.array(video_array)/255
    return video_array



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
    all_data = read_video(opt.test_dir)
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    video_lenth = len(all_data)
    for index_frame in range(video_lenth):
        source_image_base = all_data[index_frame:index_frame+1]
        source_image = source_image_base.transpose(0,3,1,2)
        source_image = torch.from_numpy(source_image)
        source_image = source_image.float()
        source_image = torch.nn.functional.interpolate(source_image,(128,128)).cuda()
        out = model.module.test(source_image)

        full_mask_bin = []
        mask_bin = out['pred_image_s_weight'][:,:,0,:,:]
        mask_bin = (torch.max(mask_bin, dim=1, keepdim=True)[0] == mask_bin).float()
        for i in range(mask_bin.shape[1]):
            mask_bin_part = mask_bin[0, i:(i+1)].data.cpu().repeat(3, 1, 1)
            color = np.array(colormap((i) / (mask_bin.shape[1]- 1)))[:3]
            if i==mask_bin.shape[1]-1:
                color = np.array((0, 0, 0))
            color = torch.from_numpy(color.reshape((3, 1, 1))).float()
            full_mask_bin.append(mask_bin_part * color)
            pass
        full_mask =sum(full_mask_bin)
        # print(q.shape)
        full_mask_b = full_mask.permute(1,2,0).detach().cpu().numpy()
        full_mask = (full_mask_b*255).astype(np.uint8)
        this_image_b = source_image_base[0]
        this_image =(this_image_b*255).astype(np.uint8)
        merge = this_image_b*0.4+full_mask_b*0.6
        merge =(merge*255).astype(np.uint8)
        final_image = res = np.hstack([this_image,full_mask,merge])
        final_image =  cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)


        cv2.imwrite(opt.out_dir+'/%04d.png'%index_frame,final_image)

       

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script")
 
    parser.add_argument("--test_dir", required=True, help="path to test dir")
    parser.add_argument("--checkpoint_path", required=True,help="path to checkpoint to restore")
    parser.add_argument("--out_dir", required=True,help="path to save_result")
    parser.add_argument("--part_numb", default=11,type=int,help="model_part_numb")
    opt = parser.parse_args()

    start_run(opt)
