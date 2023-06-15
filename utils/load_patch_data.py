import os
import sys
import math
import torch
import fnmatch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from torch.utils.data import Dataset


# 返回yolov2有关置信度的信息
from utils.n_median_pool import MedianPool2d
from vector2image import showframe_save


class yolov2_feature_output_manage(nn.Module):
    def __init__(self, cls_id, num_cls, config):
        super(yolov2_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput, loss_type):
        # get values necessary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)  # add one dimension of size 1
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)  # 其中5表示anchor的数量
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)

        # transform the output tensor from [batch, 425, 13, 13] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 169]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 169] swap 5 and 85, in position 1 and 2 respectively
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 845]
        # todo first 5 numbers that make '85' are box xc, yc, w, h and objectness. Last 80 are class prob.

        output_objectness_not_norm = output[:, 4, :]
        output_objectness_norm = torch.sigmoid(output[:, 4, :])  # [batch, 1, 845]  # iou_truth*P(obj)
        # take the fifth value, i.e. object confidence score. There is one value for each box, in total 5 boxes

        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 845]  # 845 = 5 * h * w
        # NB 80 means conditional class probabilities, one for each class related to a single box (there are 5 box for each grid cell)

        # perform softmax to normalize probabilities for object classes to [0,1] along the 1st dim of size 80 (no. classes in COCO)
        not_normal_confs = output
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # NB Softmax is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1

        # we only care for conditional probabilities of the class of interest (person, i.e. cls_id = 0 in COCO)
        confs_for_class_not_normal = not_normal_confs[:, self.cls_id, :]
        confs_for_class_normal = normal_confs[:, self.cls_id, :] # take class number 0, so just one kind of cond. prob out of 80. This is for 1 box, there are 5 boxes

        confs_if_object_not_normal = self.config.loss_target(output_objectness_not_norm, confs_for_class_not_normal)
        confs_if_object_normal = self.config.loss_target(output_objectness_norm, confs_for_class_normal)  # loss_target in patch_config

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1) # take the maximum value among your 5 priors
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.zeros(size)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf


# 返回SSD有关置信度的信息
class ssd_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    """
    def __init__(self, cls_id, num_cls, config):
        super(ssd_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, ssd_output, loss_type):
        # get values necessary for transformation
        conf_normal, conf_not_normal, loc = ssd_output
        #print('conf_normal: ' + str(conf_normal.size()))
        #print('conf_not_normal: ' + str(conf_not_normal.size()))

        # Objectness score ~= 1 - P[Background]
        # Class score = P[Class] / sum over all classes but background P[i]

        # taking results already softmaxed:
        obj_scores = torch.sum(conf_normal[:,:,1:], dim=2)
        confs_if_object_normal = conf_normal[:, :, self.cls_id] # softmaxed #obj*cls
        person_cls_score = confs_if_object_normal / obj_scores

        # taking results not softmaxed:
        # background_score_norm = torch.sigmoid(conf_not_normal[:,:,0])
        # obj_score = 1 - background_score_norm
        # cond_class_prob = torch.nn.Softmax(dim=2)(conf_not_normal[:,:,1:])
        # person_cls_score = cond_class_prob[:,:,14]

        loss_target = obj_scores
        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(loss_target, dim=1) # take the maximum value among your 5 priors
            print(max_conf)
            return max_conf
        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('ssd batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.zeros(size)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf


# 返回yolov3有关置信度的信息
class yolov3_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    """
    def __init__(self, cls_id, num_cls, config):
        super(yolov3_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        #self.num_priors = num_priors

    def forward(self, yv3_output, loss_type):
        # get values necessary for transformation
        yolo_output = yv3_output[0]
        loc = yolo_output[:, :, :4]
        objectness = yolo_output[:, :, 4]
        cond_prob = yolo_output[:, :, 5:]
        cond_prob_targeted_class = cond_prob[:, :, self.cls_id]
        confs_if_object_normal = self.config.loss_target(objectness, cond_prob_targeted_class)

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1)
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('ssd batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.zeros(size)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)
            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf


# 返回yolov4有关置信度的信息
class yolov4_feature_output_manage(nn.Module):
    def __init__(self, cls_id, num_cls, config):
        super(yolov4_feature_output_manage, self).__init__()
        self.cls_id = cls_id    # 拟隐藏类别的id
        self.num_cls = num_cls  # 识别模型总共可以识别的类数
        self.config = config    # 配置文件

    def forward(self, YOLOoutput, loss_type):
        output_list = []
        # yolo输出的三个尺度，将其转为batch*（5+cls_num）*(3*h*w) 3表示anchor的种类数
        for out_layer in YOLOoutput:
            if out_layer.dim() == 3:
                out_layer = out_layer.unsqueeze(0)  # add one dimension of size 1 if there is not
            batch = out_layer.size(0)
            assert (out_layer.size(1) == (5 + self.num_cls) * 3)
            h = out_layer.size(2)
            w = out_layer.size(3)
            output_layer = out_layer.view(batch, 3, 5 + self.num_cls, h * w)
            output_layer = output_layer.transpose(1, 2).contiguous()
            output_layer = output_layer.view(batch, 5 + self.num_cls, 3 * h * w)
            output_list.append(output_layer)
        # 输出按照特征点和到一起
        total_out = torch.cat([output_list[0], output_list[1], output_list[2]], dim=2)
        objectness_score = torch.sigmoid(total_out[:,4,:])    # 预测置信度得分
        class_cond_prob = torch.nn.Softmax(dim=1)(total_out[:, 5:5+self.num_cls,:])   # 类别信息处的softmax操作
        person_cond_prob = class_cond_prob[:, self.cls_id,:]  # 指定类别处的置信度
        confs_if_object_normal = self.config.loss_target(objectness_score, person_cond_prob)  # 根据配置文件选择检测攻击损失函数

        # 损失函数类型
        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1)  # 攻击置信度最高的目标
            return max_conf
        elif loss_type == 'threshold_approach':
            # 攻击置信度大于一定阈值的预测
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.zeros(size).cuda()   # YHQ .cuda()
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)
            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)       # YHQ 修改 torch.sum
            return thresholded_conf
        elif loss_type == 'carlini_wagner':
            loss = (objectness_score*person_cond_prob - (1 - objectness_score))
            batch_stack = torch.unbind(loss, dim=0)
            loss_total = []
            for img_loss in batch_stack:
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                batch_loss = torch.max((img_loss), zero_tensor)
                loss_total.append(batch_loss)
            loss_total = torch.stack(loss_total, 0)
            c_w_conf = torch.sum(loss_total, dim=1)
        elif loss_type == 'YAN_loss':
            yan_tmp = torch.zeros_like(objectness_score)  # 用于替换掉非目标候选区域的置信度得分
            yan_objectness = torch.where(person_cond_prob < 0.3, yan_tmp, objectness_score) # 获得所有有效的置信度  
            max_conf, _ = torch.max(yan_objectness, dim=1)  # 攻击置信度最高的目标
            return max_conf


# 返回yolov5有关置信度的信息
class yolov5_feature_output_manage(nn.Module):
    def __init__(self, cls_id, num_cls, config):
        super(yolov5_feature_output_manage, self).__init__()
        self.cls_id = cls_id    # 拟隐藏类别的id
        self.num_cls = num_cls  # 识别模型总共可以识别的类数
        self.config = config    # 配置文件

    def forward(self, YOLOoutput, loss_type):
        # print('YOLOoutput',YOLOoutput.shape) # [batch,x,6]
        # raise
        YOLOoutput = YOLOoutput[0]
        objectness_score = YOLOoutput[:, :, 4]   # 预测置信度得分
        class_cond_prob = YOLOoutput[:, :, 5:5+self.num_cls]   # 类别信息处的softmax操作
        person_cond_prob = class_cond_prob[:, :, self.cls_id]  # 指定类别处的置信度
        # print('objectness_score',objectness_score.shape, person_cond_prob.shape)
        confs_if_object_normal = self.config.loss_target(objectness_score, person_cond_prob)  # 根据配置文件选择检测攻击损失函数
        # print('confs_if_object_normal shape',confs_if_object_normal.shape)

        # 损失函数类型
        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1)  # 攻击置信度最高的目标
            # print(torch.max(max_conf.shape)
            return max_conf
        elif loss_type == 'YAN_loss':
            yan_tmp = torch.zeros_like(objectness_score)  # 用于替换掉非目标候选区域的置信度得分
            yan_objectness = torch.where(person_cond_prob < 0.1, yan_tmp, objectness_score)  # 获得属于该类别所有有效的置信度
            max_conf, _ = torch.max(yan_objectness, dim=1)  # 攻击置信度最高的目标
            return max_conf


# 计算一个patch不可以打印的可能性
class NPSCalculator(nn.Module):
    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        # 获取能打印的颜色
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # 计算Patch中颜色和能打印颜色的欧式距离
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2  # squared difference
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        color_dist_prod = torch.min(color_dist, 0)[0]
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)  # divide by the total number of elements in the input tensor

    # 读取不能打印的颜色，并将其保存在list中
    def get_printability_array(self, printability_file, side):
        printability_list = []
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)
        printability_array = np.asarray(printability_array)  # convert input lists, tuples etc. to array
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)  # Creates a Tensor from a numpy array.
        return pa


# 计算一个patch的平滑程度
class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)  # NB -1 indicates the last element!
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


# patch的转换操作（增加亮度、对比度、随机加噪声）
class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8      # 最小对比度
        self.max_contrast = 1.2      # 最大对比度
        self.min_brightness = -0.1   # 亮度变化
        self.max_brightness = 0.1    # 亮度变化
        self.noise_factor = 0.10     # 噪声幅度
        self.minangle = -20 / 180 * math.pi  # 最小旋转角度
        self.maxangle = 20 / 180 * math.pi   # 最大旋转角度
        self.medianpooler = MedianPool2d(7, same=True)  # 中值滤波

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        use_cuda = 1


        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))  # pre-processing on the image with 1 more dimension: 1 x 3 x 300 x 300, see median_pool.py

        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2  # img_size = 416, adv_patch size = patch_size in adv_examples.py, = 300
        # print('pad =' + str(pad)) # pad = 0.5*(416 - 300) = 58

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        # print('adv_patch in load_data.py, PatchTransforme, size =' + str(adv_patch.size()))
        # adv_patch in load_data.py, PatchTransforme, size =torch.Size([1, 1, 3, 300, 300]), tot 5 dimensions
        # 一张图中存在多个人
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        # print('adv_batch in load_data.py, PatchTransforme, size =' + str(adv_batch.size()))
        # adv_batch in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        # Create random contrast tensor
        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            # Fills self tensor (here 6 x 14) with numbers sampled from the continuous uniform distribution: 1/(max_contrast - min_contrast)

        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast

        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
       # 对patch进行随机增强
        adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # Consider just the first 'column' of lab_batch, where we can


        #print(cls_ids.size())                      # discriminate between detected person (or 'yes person') and 'no person')
                                                    # in this way, sensible data about x, y, w and h of the rectangles are not used for building the mask
        # NB torch.narrow returns a new tensor that is a narrowed version of input tensor. The dimension dim is input from start to start + length.
        # The returned tensor and input tensor share the same underlying storage.
        cls_mask = cls_ids.expand(-1, -1, 3)
        #print(cls_mask.size())
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # 6 x 14 x 3 x 300 x 300

        if use_cuda:
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1)
            # msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  #改
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(100) - cls_mask  # take a matrix of 1s, subtract that of the labels so that
                                                                                # we can have 0s where there is no person detected,
                                                                                # obtained by doing 1-1=0

        # NB! Now the mask has 1s 'above', where the labels data are sensible since they represent detected persons, and 0s where there are no detections
        # In this way, multiplying the adv_batch to this mask, built from the lab_batch tensor, allows to target only detected persons and nothing else,
        # i.e. pad with zeros the rest
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        # showframe_save(adv_batch.squeeze().cpu().numpy(), adv_batch.size(1), 'p')
        adv_batch = mypad(adv_batch)  # dim 6 x 14 x 3 x 416 x 416
        # showframe_save(adv_batch.squeeze().cpu().numpy(), adv_batch.size(1), 'p')
        msk_batch = mypad(msk_batch)  # dim 6 x 14 x 3 x 416 x 416
        # 随机旋转角度的生成
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # dim = 6*14 = 84
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.FloatTensor(anglesize).fill_(0)
        # 缩放和旋转
        current_patch_size = adv_patch.size(-1)  # 300
        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)  # dim 6 x 14 x 5  #读
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        # 计算预计的patch大小
        # mul(0.2)
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.25)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.25)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # np.prod(batch_size) = 4*16 = 84
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # used to get off_x
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # used to get off_y
        if(rand_loc):
            if use_cuda:
                off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))    # (-0.4,0.4)
                off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))   # (-0.4,0.4)
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))   # (-0.4,0.4)
                off_y = targetoff_y * (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))   # (-0.4,0.4)
            target_x = target_x + off_x
            target_y = target_y + off_y
        target_y = target_y - 0.01      # 0.05太大
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)
        s = adv_batch.size() # 6 x 14 x 3 x 416 x 416
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16
        tx = (-target_x+0.5)*2   # 0.5
        ty = (-target_y+0.5)*2   # 0.5
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        # Theta = rotation, rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0) # dim 84 x 2 x 3 (N x 2 x 3) required by F.affine_grid

        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2
        # showframe_save(adv_batch.squeeze().cpu().numpy(), adv_batch.size(1), 'p')
        adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg
        # showframe_save(adv_batch.squeeze().cpu().numpy(), adv_batch.size(1), 'p')
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) # 4 x 16 x 3 x 416 x 416
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        return adv_batch_t * msk_batch_t  # It is as if I have passed adv_batch_t "filtered" by the mask itself


# 将patch添加到图片
class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):

        advs = torch.unbind(adv_batch, 1)  # Returns a tuple of all slices along a given dimension, already without it.
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)  # the output tensor has elements belonging to img_batch if adv == 0, else belonging to adv
        return img_batch


# Inria数据集
class InriaDataset(Dataset):
    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images   # 图片的数量
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        # 获取图片名称
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []  # 图片的路径
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab    # 最多的标注信息数量

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')  # 图片
        if os.path.getsize(lab_path):      # check to see if label file contains data.
            label = np.loadtxt(lab_path)   # 标注信息
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)
        image, label = self.pad_and_scale(image, label)   # 填充成正方形
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        return image, label

    # 将输入图片按照中心进行填充
    def pad_and_scale(self, img, lab):
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    # 扩充函数
    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=100)  # padding of the labels to have a pad_size = max_lab (14 here).
                                                                   # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
                                                                   # whose number is not known a priori
        else:
            padded_lab = lab
        return padded_lab