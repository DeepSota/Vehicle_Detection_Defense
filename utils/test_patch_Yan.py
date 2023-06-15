# -*- coding: utf-8 -*-
import torch
import shutil

from torchvision import transforms
import warnings, pdb

from utils.load_patch_data import *
from vector2image import showframe_save

warnings.filterwarnings("ignore")
from example.models.common import DetectMultiBackend

# 随机生成一个Patch
def generate_patch(type, patch_size):
    if type == 'gray':
        adv_patch_cpu = torch.full((3, patch_size, patch_size), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, patch_size, patch_size))
    return adv_patch_cpu





# 退填充后的图片进行裁剪，还原图片原始大小
def remove_pad(w_orig, h_orig, in_img):
        w = w_orig
        h = h_orig
        img = transforms.ToPILImage('RGB')(in_img)
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            image = Image.Image.resize(img, (h, h))
            image = Image.Image.crop(image, (int(padding), 0, int(padding) + w, h))
        else:
            padding = (w - h) / 2
            image = Image.Image.resize(img, (w, w))
            image = Image.Image.crop(image, (0, int(padding), w, int(padding) + h))
        return image



# use_cuda = 1
#
# weightfile = "coco_yolov5m.pt"
# imgdir = "dataset/train"
# # imgdir = "crawler_data/imgs"                # 测试数据集
# patchfile = "Patches/patch/yolov5_2.png"       # 对抗Patch路径
# square_size = 640  # 输入图片大小
# patch_size = 300   # 对抗Patch大小
#
# # 读取Patch，并将其转为tensor
# patch_img = Image.open(patchfile).convert('RGB')
# patch_img = patch_img.resize((patch_size, patch_size))
# adv_patch = transforms.ToTensor()(patch_img)
#
# # 检测模型的设置
# # yolov5 = torch.hub.load('ultralytics/yolov5','crawl_yolo',pretrained = True, verbose=False).eval()
# device ='cuda:0'
# yolov5 = DetectMultiBackend(weightfile)
# for m in yolov5.modules():
#     if hasattr(m, 'inplace'):
#         m.inplace = False
# yolov5 = yolov5.eval().to(device)
#
# yolov5.classes = [0]
# yolov5.conf = 0.7
# m = yolov5
# if use_cuda:
#     adv_patch = adv_patch.cuda()
# save_path = 'save_results/0104'
# label_dir='dataset/labels'
# i = 0
# for imgfile in os.listdir(imgdir):
#     print(imgfile)
#     name = os.path.splitext(imgfile)[0]                          # 图片名
#     img_path = os.path.join(imgdir, imgfile)
#     txt_clean_name = imgfile.replace('.png','.txt')              # 标注信息
#     patch_label_name = os.path.splitext(imgfile)[0] + '_p.txt'
#
#     # txt_clean_path = os.path.join('save_results/clean/', 'yolov4-labels/', txt_clean_name)
#     txt_clean_path = os.path.join(label_dir, txt_clean_name)
#     img = Image.open(img_path).convert('RGB')
#
#     h_orig = img.size[1]  # 输入图片的宽
#     w_orig = img.size[0]  # 输入图片的长
#     # 读取图片的标注信息，一方面用于图像的填充，一方面用于Patch贴图
#     textfile = open(txt_clean_path, 'r')
#     if os.path.getsize(txt_clean_path):  # check to see if label file contains data.
#         label = np.loadtxt(textfile)
#     else:
#         label = np.ones([5])
#     if np.ndim(label) == 1:
#         label = np.expand_dims(label, 0)
#     label = torch.from_numpy(label).float()
#     image_clean_ref = img
#     # 填充和缩放
#     image_p, label = pad_and_scale(image_clean_ref, label, common_size=square_size)
#     image_tens = transforms.ToTensor()(image_p)
#     img_fake_batch = torch.unsqueeze(image_tens, 0).cuda()
#     lab_fake_batch = torch.unsqueeze(label, 0).cuda()
#     # Patch转换增广操作
#     adv_batch_t = PatchTransformer()(adv_patch, lab_fake_batch, square_size, do_rotate=True, rand_loc=False)
#     # 将Patch贴到目标
#     p_img_batch = PatchApplier()(img_fake_batch, adv_batch_t)
#     import matplotlib.pyplot as plt
#     image = p_img_batch.cpu().numpy()
#     showframe_save(image, 1,  name)
#
#
#
#
#     # p_img = torch.squeeze(p_img_batch, 0)
#     # 贴Patch后产生的图片
#     # pdb.set_trace()
#
#     # p_img_NT = transforms.ToPILImage('RGB')(p_img_batch[0])
#     # pdb.set_trace()
#     results = m(p_img_batch)[0]
#     #results.save()                                          # 保存检测结果到runs/detect
#     # 将runs/detect/目录的检测可视化结果移到savedImage
#     if i < 1:
#         img_path = 'runs/detect/exp/image0.jpg'
#     else:
#         img_path = 'runs/detect/exp{}/image0.jpg'.format(i+1)
#     i = i+1
#     # shutil.move(img_path, os.path.join(save_path,'{}.png'.format(name)))
# # shutil.rmtree('runs/detect')