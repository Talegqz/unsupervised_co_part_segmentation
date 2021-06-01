import torch
import torch.nn as nn
from torch.nn import parallel
from networks  import Hourglass,UpBlock2d,ResBlock2d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.functional as F
import numpy as np
from torchvision import models

# rotation scale theta
def get_affine_inverse(affine_matrix):
    R = affine_matrix[:,0:2,0:2]
    t = affine_matrix[:,0:2,2:3].view(-1,2,1)
    R_inverse = torch.inverse(R)
    t_inverse = -torch.bmm(R_inverse,t).view(-1,2,1)

    return torch.cat([R_inverse,t_inverse],dim=2)
    pass


def get_affine_matrix_with_cossin(cos_sin,scale,shift):
    scale = torch.diag_embed(scale)
    
    cos_theta = cos_sin[:,0:1]
    sin_theta = cos_sin[:,1:2]
    rotate = torch.cat((cos_theta, -sin_theta, sin_theta, cos_theta),dim=-1).view(-1,2,2)
    rotate_scale = torch.bmm(rotate, scale)
    shift = shift.view(-1,2,1)
    affine = torch.cat((rotate_scale, shift), dim=-1)
    return affine


def get_mu_and_cov(part_maps,thr = 0.02):
    # 计算平均值 和协方差矩阵的函数
    # count the value >0.1  in w h channel
    batch,channels,w,h =part_maps.shape
    count_value = (part_maps>thr).view(batch,channels,-1).sum(dim=-1).type(part_maps.type())+1
    # print(count_value)
    # print(count_value.shape,'count_value.shape')

    # 生成grid
    x_t = torch.linspace(-1.0, 1.0, w).reshape((w,1)).repeat(1,h).unsqueeze(0).type(part_maps.type())
    y_t = torch.linspace(-1.0, 1.0, 
    h).reshape((1,h)).repeat(w,1).unsqueeze(0).type(part_maps.type())
    meshgrid = torch.cat([y_t,x_t],0).type(part_maps.type()).view(1,1,2,w,h)
    part_maps_mask = part_maps.unsqueeze(2).clone()
    part_maps_mask[part_maps_mask<=thr] = 0
    # print(meshgrid.shape,"meshgrid.shape")
    # print(part_maps.shape,"part_maps.shape")
    # 激活值和网格相乘
    mesh_value = part_maps_mask*meshgrid
    # print(mesh_value.shape,"mesh_value.shape")
    # get mu
    mu = mesh_value.view(batch,channels,2,-1).sum(dim=-1)/(count_value.view(batch,channels,1))
    # print(mu.shape,"mu.shape")
    
    mu_out_prod = mu.view(batch,channels,1,2).repeat(1,1,2,1)* mu.view(batch,channels,2,1)
    mesh_out_part_value = (mesh_value.view(batch,channels,1,2,-1).repeat(1,1,2,1,1)*mesh_value.view(batch,channels,2,1,-1)).sum(dim=-1)/(count_value.view(batch,channels,1,1))

    # # 得到协方差矩阵
    
    cov = mesh_out_part_value-mu_out_prod
    # return 0,0,0
    return cov


def get_coordinate_tensors(x_max, y_max,this_type):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).type(this_type)
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).type(this_type)

    return x_map_tensor, y_map_tensor


def volume_center_loss(volume):
    batch,part,d,w,h = volume.shape
    # volume = volume.clone()
    grid =  make_coordinate_grid_3d((d,w,h),volume.type())
    loss = 0
    for b in range(batch):
        for p in range(part):
            this_volume = volume[b][p].unsqueeze(3)
            out = this_volume*grid
            average = (out/(this_volume.sum()+1e-10)).view(-1,3).sum(dim=0)
            # print(average)
            loss +=torch.abs(average).mean()
    
    
    # print(all_average)
    return loss/(batch*part)


def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w,part_map.type())

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-8, self_ref_coord=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] 
        k = (part_map).sum()+epsilon
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)


def get_variance(part_map, x_c, y_c):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w,part_map.type())

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y


def batch_get_centers(pred_softmax):
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)



def concentration_loss(pred_softmax,zero_center=False):
    
    B,part_numb,C,H,W = pred_softmax.shape
    pred_softmax = pred_softmax.view(B,part_numb,H,W)
    pred_softmax = pred_softmax[:,0:-1]
    loss = 0
    epsilon = 1e-8
    centers_all = batch_get_centers(pred_softmax)
    if zero_center:
        # return torch.abs(centers_all).mean().sum()
        centers_all =  torch.zeros(centers_all.shape).type(centers_all.type())
        
    for b in range(B):
        centers = centers_all[b]
        for c in range(part_numb-1):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:]  # prevent gradient explosion
            k = (part_map).sum() + epsilon
            part_map_pdf = part_map/k
            x_c, y_c = centers[c]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
    return loss/B


def generate_invese_matrix_for_torch(theta,scale, shift):
    inv_scale = torch.diag_embed(1. / scale)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    inv_rotate = torch.cat((cos_theta, sin_theta, -sin_theta, cos_theta),dim=-1).view(-1,2,2)

    inv_scale_rotate = torch.bmm(inv_scale, inv_rotate)

    inv_shift = torch.bmm(inv_scale_rotate, -shift.view(-1,2,1))
    inv_affine = torch.cat((inv_scale_rotate, inv_shift), dim=-1)

    return inv_affine


class Feat_To_Camera(nn.Module):
    def __init__(self,part_numb=5):
        super(Feat_To_Camera, self).__init__()

        # 这里输入的shape 是  16,32 32 32
        # 这里要得到 K 个 part 这里是16
        self.part_numb = part_numb
        self.volume_feature = nn.Sequential(
            nn.Conv2d(part_numb*4*32, part_numb*32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= part_numb*32,
                    out_channels=part_numb*16, kernel_size=4, stride=2,padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=part_numb*16,
                    out_channels=part_numb*4, kernel_size=4, stride=2,padding=1)
            
        )

        self.get_joint = nn.Linear(part_numb*4*4*4,(self.part_numb-1)*2)
        # self.get_joint.weight.data.zero_()
        self.get_joint.bias.data.copy_(torch.tensor([1,0] * (part_numb-1), dtype=torch.float))
        self.get_shift = nn.Linear(part_numb*4*4*4,(self.part_numb-1)*2)
        self.get_shift.weight.data.zero_()
        self.get_shift.bias.data.zero_()

        self.get_scale = nn.Linear(part_numb*4*4*4,(self.part_numb-1)*2)
        self.get_scale.weight.data.zero_()
        self.get_scale.bias.data.zero_()
        
    def forward(self, x):
        feature = self.volume_feature(x)
        b,t,w,h = feature.shape
    
        feature = feature.view(b, -1)
        # affine = affine.sum(-1)
        zeros_affine = torch.tensor([1,0]*b).view(b,1,2).type(x.type())
        
        affine = self.get_joint(feature)
        affine = affine.view(b,-1,2)
        # print(affine)
        # affine = torch.tanh(affine)*np.pi
        affine  = torch.cat([affine,zeros_affine],dim=1)
        affine = affine.view(-1,  2)
        affine = affine/(torch.norm(affine,p=2,dim=1,keepdim=True)+1e-10)
        # print(affine)
        # 
        # print(affine)
        shift = self.get_shift(feature)
        zeros_shift = torch.zeros((b,1,2)).type(shift.type())
        shift = shift.view(-1, self.part_numb-1, 2)
        # shift = torch.tanh(shift)
        shift = torch.cat([shift,zeros_shift],dim=1)
        shift = shift.view(-1,2,1)
        # print(shift)


        scale = self.get_scale(feature)
        zeros_scale = torch.ones((b,1,2)).type(shift.type())
        scale = scale.view(-1, self.part_numb-1, 2)
        scale = torch.tanh(scale)*0.9+1
        # scale = torch.clamp(scale,min=0.1,max=10)
        scale = 1/scale
        scale = torch.cat([scale,zeros_scale],dim=1)
        scale = scale.view(-1,2)

        theta = affine.view(b*self.part_numb,2)
        shift = shift.view(b*self.part_numb,2)
        scale = scale.view(b*self.part_numb,2)
        
        return theta,scale,shift
        # return affine_f


class Encoder(nn.Module):
    def __init__(self,part_numb=10):
        super(Encoder, self).__init__()
        self.base_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ELU(inplace=True),
        )
        self.hourglas = Hourglass(512, 512, 3, 1024)
        self.merge_feature = nn.Sequential(
            nn.Conv2d(1024, 32*part_numb*4, 3, 1, 1, bias=True),
        )  
        self.get_2d_heat = nn.Sequential(
            nn.Conv2d(32*part_numb*4, part_numb*32, 7, 1, 3, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(part_numb*32, part_numb*10, 7, 1, 3, bias=True)
        )

        self.part_numb = part_numb
        self.get_affine = Feat_To_Camera(part_numb)
        # self.volume_feature = nn.ModuleList()

    def forward(self, input_image):
        b =input_image.shape[0]
        feature = self.base_conv(input_image)
        feature = self.hourglas(feature)
        merge_feature = self.merge_feature(feature)

       
        volume = self.get_2d_heat(merge_feature)

        volume = volume.view(b,self.part_numb,-1,32,32)
        theta,scale,shift = self.get_affine(merge_feature)




        return volume,theta,scale,shift

class Decoder(nn.Module):
    def __init__(self,part_numb=10):
        super(Decoder, self).__init__()

        self.part_numb = part_numb

        self.get_image_para = nn.Sequential(
                nn.Conv2d(10,64,7,1,3),
                nn.LeakyReLU(0.2),
                ResBlock2d(64,3,1),
                nn.Conv2d(64,128,7,1,3),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128,128,3,1,1),
                nn.LeakyReLU(0.2),
                UpBlock2d(128,64),
                UpBlock2d(64,32),
                nn.Conv2d(32,16,3,1,1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(16,4,3,1,1),
                # nn.LeakyReLU(0.2)
                # nn.Sigmoid()
            )

    def forward(self, x):
        all_image  = []
        all_weight = []
        # b = x.shape[0]
        for i in range(self.part_numb):
            # print(x.shape)
            out_feat = self.get_image_para(x[:,i])
            # print(out_feat.shape)
            all_image.append(out_feat[:,0:3])
            all_weight.append(out_feat[:,3:4])
        all_image = torch.stack(all_image,dim=1)
        all_weight = torch.stack(all_weight,dim=1)
        b,p,c,w,h = all_weight.shape
        # all_weight = F.tanh(all_weight)
        all_weight = all_weight
        all_weight = all_weight.view(b,p,-1)
        all_weight = F.softmax(all_weight,dim=1)
        all_weight = all_weight.view(b,p,c,w,h)


        fina_image = all_image*all_weight
        fina_image = torch.sum(fina_image,dim=1)
        # image = torch.sum(s)
        # print(all_image.shape)
        return fina_image,all_image,all_weight

 
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

vgg = Vgg19().cuda()

def get_vgg_loss(x_vgg, y_vgg):
    value_total = 0
    loss_weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  
    for i in range(len(x_vgg)):
        b, c, w, h = x_vgg[i].shape
        # print('vggggggggg',x_vgg[i].shape)
        feature_diff = (x_vgg[i] - y_vgg[i].detach())
        value = torch.abs(feature_diff).mean()
        value_total += value*loss_weight[i]
    return value_total
def get_random_matrix(batch_size,data_type):
    # scale = torch.rand(1)*0.25+0.75
    scale = torch.rand(batch_size,2)*0.25+0.75
    rotation_joint = torch.rand(batch_size,1)*np.pi*2
    # print(rotation_joint)
    shift=torch.rand(batch_size,2)*0.2
    basic = generate_invese_matrix_for_torch(rotation_joint,scale,shift).type(data_type)
    
    return basic

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)


class Part_3D_Disnet(nn.Module):
    def __init__(self,part_numb=16):
        super(Part_3D_Disnet, self).__init__()
        self.encoder = Encoder(part_numb)
        self.decoder = Decoder(part_numb)

        self.part_numb=part_numb
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        


    def affine_trans_image(self,image):
        b = image.shape[0]
        affine_matrix = get_random_matrix(b,image.type())
        # affine_matrix = affine_matrix.repeat(image.shape[0],1,1)
        grid = F.affine_grid(affine_matrix,image.size())
        out_image = F.grid_sample(image,grid,padding_mode='border')
        del grid
        return out_image,affine_matrix

    def transform_volume(self,volume,affine):
        b, part_numb, d, w, h = volume.shape
        s_volume = volume.view(b*part_numb, d, w, h)
        affine_grid_s = F.affine_grid(affine, s_volume.shape)
        s_volume = F.grid_sample(s_volume, affine_grid_s)
        s_volume = s_volume.view(b, part_numb, d, w, h)
        return s_volume
    def test(self, image_source):
        out = {}
        s_volume,s_theta,s_scale,s_shift = self.encoder(image_source)
  
        b, part_numb, d, w, h = s_volume.shape
        
        s_affine_to_latent = get_affine_matrix_with_cossin(s_theta,s_scale,s_shift)
        s_affine_for_torch = get_affine_inverse(s_affine_to_latent)
        
        s_latent_volume = self.transform_volume(s_volume,s_affine_to_latent)

        s_back_volume = self.transform_volume(s_latent_volume,s_affine_for_torch)


        pred_image_s,pred_image_s_part,pred_image_s_weight = self.decoder(s_back_volume)

        
        out['pred_image_s'] = pred_image_s
        out['pred_image_s_part'] = pred_image_s_part
        out['pred_image_s_weight'] = pred_image_s_weight

        return out


    def forward(self, image_source, image_target):
        out = {}
        s_volume,s_theta,s_scale,s_shift = self.encoder(image_source)

        t_volume,t_theta,t_scale,t_shift = self.encoder(image_target)
        
        

        b, part_numb, d, w, h = s_volume.shape

        out['t_shift'] = t_shift.view(b,part_numb,2)
        out['s_shift'] = s_shift.view(b,part_numb,2)
        
        s_affine_to_latent = get_affine_matrix_with_cossin(s_theta,s_scale,s_shift)
        s_affine_for_torch = get_affine_inverse(s_affine_to_latent)

        t_affine_to_latent = get_affine_matrix_with_cossin(t_theta,t_scale,t_shift)
        t_affine_for_torch = get_affine_inverse(t_affine_to_latent)
        
        
        s_latent_volume = self.transform_volume(s_volume,s_affine_to_latent)
        t_latent_volume = self.transform_volume(t_volume,t_affine_to_latent)
        
        
        s_warp_to_t_volume = self.transform_volume(s_latent_volume,t_affine_for_torch)
        t_warp_to_s_volume = self.transform_volume(t_latent_volume,s_affine_for_torch)

        s_back_volume = self.transform_volume(s_latent_volume,s_affine_for_torch)

        out['t_affine'] = t_affine_for_torch.view(b, part_numb, 2, 3)
        out['s_affine'] = s_affine_for_torch.view(b, part_numb, 2, 3)

        out['t_affine_left'] = t_affine_to_latent.view(b, part_numb, 2, 3)
        out['s_affine_left'] = s_affine_to_latent.view(b, part_numb, 2, 3)
    
        out['s_warp_to_t_volume'] = s_warp_to_t_volume
        out['t_warp_to_s_volume'] = t_warp_to_s_volume

        out['t_volume'] = t_volume

        out['s_volume'] = s_volume

        out['s_latent_volume'] = s_latent_volume

        out['t_latent_volume'] = t_latent_volume

        # out['t_latent_volume'] = t_latent_volume

        pred_image_t,pred_image_t_part,pred_image_t_weight = self.decoder(s_warp_to_t_volume)

        pred_image_s,pred_image_s_part,pred_image_s_weight = self.decoder(s_back_volume)

        pred_latent_s,pred_latent_part,pred_latent_weight = self.decoder(s_latent_volume)

        out['pred_image_t'] = pred_image_t
        out['pred_image_t_part'] = pred_image_t_part
        out['pred_image_t_weight'] = pred_image_t_weight
        
        out['pred_image_s'] = pred_image_s
        out['pred_image_s_part'] = pred_image_s_part
        out['pred_image_s_weight'] = pred_image_s_weight
 

        out['pred_latent_s'] = pred_latent_s
        out['pred_latent_weight'] = pred_latent_weight
        out['pred_latent_part'] = pred_latent_part
        out['image_source'] = image_source
        out['image_target'] = image_target
    
        image_target_warp,warp_matrix = self.affine_trans_image(image_target)
        t_affine_base_volume,warp_theta,warp_scale,warp_shift = self.encoder(image_target_warp)

        t_affine_warp_to_latent = get_affine_matrix_with_cossin(warp_theta,warp_scale,warp_shift)
        t_affine_warp= get_affine_inverse(t_affine_warp_to_latent)

        out['image_target_warp'] = image_target_warp
        out['t_affine_base_volume'] = t_affine_base_volume
        out['t_affine_warp'] = t_affine_warp
        out['t_affine_warp_to_latent'] = t_affine_warp_to_latent

        t_affine_base_volume_from_latent = self.transform_volume(s_latent_volume,t_affine_warp)
        out['t_affine_base_volume_from_latent'] = t_affine_base_volume_from_latent

        warp_matrix = warp_matrix.view(b,1,2,3).repeat(1,part_numb,1,1).view(b*part_numb,2,3)
 
        t_affine_volume = self.transform_volume(s_warp_to_t_volume,warp_matrix)
  

        out['t_affine_volume'] = t_affine_volume

        pred_image_warp,pred_image_warp_part,pred_image_warp_weight = self.decoder(t_affine_volume)
        out['pred_image_warp'] = pred_image_warp
        out['pred_image_warp_weight'] = pred_image_warp_weight
        out['pred_image_warp_part'] = pred_image_warp_part
        out['warp_matrix'] = warp_matrix.view(b, part_numb, 2, 3)

        return out
    


    def calculate_loss(self, out,arg):
        loss_dict = {}
  

        b = out['pred_image_t'].shape[0]
        p = self.part_numb
        if arg.vgg_loss:
            xvgg = vgg(out['pred_image_t'])
            yvgg = vgg(out['image_target'])
            loss_dict['loss_t'] = get_vgg_loss(xvgg, yvgg)*arg.vgg_weight
            loss_dict['loss_t_mse'] = self.l1(out['pred_image_t'], out['image_target'])
            loss_dict['loss_t_gradient'] = gradient_loss(out['pred_image_t'], out['image_target'])*arg.gradient_weight

        if arg.vgg_loss_source:
            pred = out['pred_image_s']
            target = out['image_source']
            xvgg = vgg(pred)
            yvgg = vgg(target)
            loss_dict['loss_s'] = get_vgg_loss(xvgg, yvgg)*arg.vgg_weight *arg.weight_s_image
            loss_dict['loss_s_mse'] = self.l1(pred, target)*arg.weight_s_image
            loss_dict['loss_s_gradient'] = gradient_loss(pred, target)*arg.gradient_weight*arg.weight_s_image

        if arg.tran_flag:
            out['t_shift'] = out['t_shift'].view(b,p,2)
            shift_value = out['t_shift'][:,0:-1]
            part_center = batch_get_centers(out['pred_image_t_weight'][:,:-1,0])
            loss_dict['shift_loss'] = self.l1(shift_value,part_center)*arg.weight_tran
        if arg.latent_center:
            loss_dict['concentration_center_loss']=  concentration_loss(out['pred_latent_weight'],True)*arg.weight_latent_center
        if arg.con_flag:
            loss_dict['concentration_loss']=  concentration_loss(out['pred_image_t_weight'])*arg.weight_con
            loss_dict['concentration_loss_s']=  concentration_loss(out['pred_image_s_weight'])*arg.weight_con
        if arg.mask_flag:
            mask_part = out['pred_image_t_weight'][:,-1]
            loss_dict['mask_back_ground_loss']=  self.l1(mask_part,torch.ones(mask_part.shape).type(mask_part.type()))*arg.weight_mask
            mask_part = out['pred_image_t_weight'][:,0:-1]
            # mask_part = mask_part.sum()
            loss_dict['mask_part_zero_loss']=  self.l1(mask_part,torch.zeros(mask_part.shape).type(mask_part.type()))*arg.weight_mask_zero
            # mask_part = out['pred_image_t_weight'][:,[2,4,8]]
            # loss_dict['mask_loss2']=  self.l1(mask_part,torch.zeros(mask_part.shape).type(mask_part.type()))*15
        # if equivariance_affine_flag:
        if arg.eq_flag:
            b,part_numb,_,_,_ = out['s_volume'].shape
            used_matrix = out['warp_matrix'].view(b,part_numb,2,3)[:,0:-1].contiguous().view(b*(part_numb-1),2,3)
            pred_matrix = out['t_affine_warp'].view(b,part_numb,2,3)[:,0:-1].contiguous().view(b*(part_numb-1),2,3)
            pred_matrix_basic = out['t_affine'].view(b,part_numb,2,3)[:,0:-1].contiguous().view(b*(part_numb-1),2,3)
            used_for_inverse = ((torch.Tensor([0, 0, 1])).unsqueeze(
                0).unsqueeze(0)).repeat(b*(part_numb-1), 1, 1).type(used_matrix.type())
        
            pred_matrix = torch.cat([pred_matrix,used_for_inverse],dim=1)
            
            used_matrix = torch.cat([used_matrix,used_for_inverse],dim=1)

            pred_matrix_basic = torch.cat([pred_matrix_basic,used_for_inverse],dim=1)
            # used_matrix = get_affine_inverse(used_matrix)
            # pred_matrix = get_affine_inverse(pred_matrix)
            # pred_matrix_basic = get_affine_inverse(pred_matrix_basic)

            value = torch.bmm(pred_matrix_basic,used_matrix )
            loss = torch.abs(pred_matrix - value).mean()
            # value = torch.bmm(inverse_pred_matrix,value )

            # eye = torch.eye(4).view(1,1,4,4).type(value.type())
            # value = torch.abs(eye - value).mean()
            loss_dict['equivariance_affine'] = loss*arg.weight_eq
        if   "rots_flag" in arg and arg.matrix_flag:
            # first shift the part to center
            we_need_part = out['pred_image_t_weight'][:,:-1,0]
            b,p,w,h = we_need_part.shape
            out['t_shift'] = out['t_shift'].view(b,p+1,2)
            center = out['t_shift'][:,0:-1]*(-1)
            we_need_part =we_need_part.contiguous().view(b*p,1,w,h)
            center = center.view(b*p,2,-1)
            affine = torch.zeros((b*p,2,2)).type(center.type())
            affine[:,0,0]=1
            affine[:,1,1]=1

            matrix = torch.cat([affine,center],dim=2)
            grid = F.affine_grid(matrix,we_need_part.size())
            we_need_part = F.grid_sample(we_need_part,grid)
            we_need_part =we_need_part.view(b,p,w,h)
            
            batch_cov = get_mu_and_cov(we_need_part).view(-1,2,2)
            matrix_this =  out['t_affine_left'][:,:-1,:,0:2].contiguous().view(-1,2,2)
            # matrix_this_inverse = torch.inverse(matrix_this)
            loss_dict['matrix_loss'] = torch.norm(torch.bmm(matrix_this,matrix_this.permute((0,2,1)))-batch_cov,p=2)*arg.weight_rots
            

            pass
        return loss_dict 
        # pass

if __name__ == "__main__":
    pass
