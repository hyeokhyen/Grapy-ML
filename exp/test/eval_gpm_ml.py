import timeit
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML')
import json

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

from dataloaders import cihp, atr, pascal
from networks import graph, grapy_net
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_, eval_with_numpy
from easydict import EasyDict as edict

sys.path.append('/nethome/hkwon64/Research/imuTube/code_video2imu_v2')
from analysis.util.vis import draw_mask, draw_skeleton
from analysis.util.bbox import bbox_expand as func_bbox_expand

classes = [
  'background', 
  'head', 
  'torso', 
  'upper-arm', 
  'lower-arm', 
  'upper-leg',
  'lower-leg']

colors = [[255, 0, 0], 
				[255, 255, 0],
				[0, 255, 0],
				[0, 255, 255], 
				[0, 0, 255], 
				[255, 0, 255]]		


'''
0: 'Background', 
1: 'Head', 
2: 'Torso', 
3: 'Upper Arms', 
4: 'Lower Arms', 
5: 'Upper Legs', 
6: 'Lower Legs'
'''


gpu_id = 0

label_colours = [(0,0,0)
        , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
        , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
  indices = [slice(None)] * x.dim()
  indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                dtype=torch.long, device=x.device)
  return x[tuple(indices)]


def flip_cihp(tail_list):
  '''

  :param tail_list: tail_list size is 1 x n_class x h x w
  :return:
  '''
  # tail_list = tail_list[0]
  tail_list_rev = [None] * 20
  for xx in range(14):
    tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
  tail_list_rev[14] = tail_list[15].unsqueeze(0)
  tail_list_rev[15] = tail_list[14].unsqueeze(0)
  tail_list_rev[16] = tail_list[17].unsqueeze(0)
  tail_list_rev[17] = tail_list[16].unsqueeze(0)
  tail_list_rev[18] = tail_list[19].unsqueeze(0)
  tail_list_rev[19] = tail_list[18].unsqueeze(0)
  return torch.cat(tail_list_rev, dim=0)


def flip_atr(tail_list):
  '''

  :param tail_list: tail_list size is 1 x n_class x h x w
  :return:
  '''
  # tail_list = tail_list[0]
  tail_list_rev = [None] * 18
  for xx in range(9):
    tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
  tail_list_rev[10] = tail_list[9].unsqueeze(0)
  tail_list_rev[9] = tail_list[10].unsqueeze(0)
  tail_list_rev[11] = tail_list[11].unsqueeze(0)
  tail_list_rev[12] = tail_list[13].unsqueeze(0)
  tail_list_rev[13] = tail_list[12].unsqueeze(0)
  tail_list_rev[14] = tail_list[15].unsqueeze(0)
  tail_list_rev[15] = tail_list[14].unsqueeze(0)
  tail_list_rev[16] = tail_list[16].unsqueeze(0)
  tail_list_rev[17] = tail_list[17].unsqueeze(0)

  return torch.cat(tail_list_rev, dim=0)

def decode_labels(mask, num_images=1, num_classes=20):
  """Decode batch of segmentation masks.

  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

  Returns:
    A batch with num_images RGB images of the same size as the input.
  """
  n, h, w = mask.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :]):
      for k_, k in enumerate(j):
        if k < num_classes:
          pixels[k_,j_] = label_colours[k]
    outputs[i] = np.array(img)
  return outputs

def get_parser():
  '''argparse begin'''
  parser = argparse.ArgumentParser()
  LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--batch', default=16, type=int)
  parser.add_argument('--lr', default=1e-7, type=float)
  parser.add_argument('--numworker', default=12, type=int)
  parser.add_argument('--step', default=30, type=int)
  # parser.add_argument('--loadmodel',default=None,type=str)
  parser.add_argument('--classes', default=7, type=int)
  parser.add_argument('--testepoch', default=10, type=int)
  parser.add_argument('--loadmodel', default='', type=str)
  parser.add_argument('--txt_file', default='', type=str)
  parser.add_argument('--hidden_layers', default=128, type=int)
  parser.add_argument('--gpus', default=4, type=int)
  parser.add_argument('--output_path', default='./results/', type=str)
  parser.add_argument('--gt_path', default='./results/', type=str)

  parser.add_argument('--resume_model', default='', type=str)

  parser.add_argument('--hidden_graph_layers', default=256, type=int)
  parser.add_argument('--dataset', default='cihp', type=str)

  opts = parser.parse_args()
  return opts


def main(opts):

  '''
  Namespace(
    batch=1, 
    classes=7, 
    dataset='pascal', 
    epochs=100, 
    gpus=1, 
    gt_path='./data/datasets/pascal/SegmentationPart/',
    hidden_graph_layers=256, 
    hidden_layers=128, 
    loadmodel='', 
    lr=1e-07, 
    numworker=12, 
    output_path='./result/gpm_ml_pascal', 
    resume_model='./data/models/GPM-ML_finetune_PASCAL.pth',
    step=30, 
    testepoch=10, 
    txt_file='./data/datasets/pascal/list/val_id.txt'
    )
  '''

  opts = edict()
  opts.batch=1 
  opts.classes=7 
  opts.dataset='pascal' 
  opts.epochs=100
  opts.gpus=1
  opts.gt_path='/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML/data/datasets/pascal/SegmentationPart/'
  opts.hidden_graph_layers=256
  opts.hidden_layers=128
  opts.loadmodel=''
  opts.lr=1e-07
  opts.numworker=12
  opts.output_path='/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML/result/gpm_ml_demo'
  opts.resume_model='/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML/data/models/GPM-ML_finetune_PASCAL.pth'
  opts.step=30
  opts.testepoch=10
  opts.txt_file='/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML/data/datasets/pascal/list/val_id.txt'

  with open(opts.txt_file, 'r') as f:
    img_list = f.readlines()

  net = grapy_net.GrapyMutualLearning(os=16, hidden_layers=opts.hidden_graph_layers)

  if gpu_id >= 0:
    net.cuda()

  if not opts.resume_model == '':
    x = torch.load(opts.resume_model)
    net.load_state_dict(x)

    print('resume model:', opts.resume_model)

  else:
    print('we are not resuming from any model')

  if opts.dataset == 'cihp':
    val = cihp.VOCSegmentation
    val_flip = cihp.VOCSegmentation

    vis_dir = '/cihp_output_vis/'
    mat_dir = '/cihp_output/'

    num_dataset_lbl = 0

  elif opts.dataset == 'pascal':

    val = pascal.VOCSegmentation
    val_flip = pascal.VOCSegmentation

    vis_dir = '/pascal_output_vis/'
    mat_dir = '/pascal_output/'

    num_dataset_lbl = 1

  elif opts.dataset == 'atr':
    val = atr.VOCSegmentation
    val_flip = atr.VOCSegmentation

    vis_dir = '/atr_output_vis/'
    mat_dir = '/atr_output/'

    print("atr_num")
    num_dataset_lbl = 2

  ## multi scale
  scale_list=[1,0.5,0.75,1.25,1.5,1.75]
  testloader_list = []
  testloader_flip_list = []
  for pv in scale_list:
    composed_transforms_ts = transforms.Compose([
      tr.Scale_(pv),
      tr.Normalize_xception_tf(),
      tr.ToTensor_()])

    composed_transforms_ts_flip = transforms.Compose([
      tr.Scale_(pv),
      tr.HorizontalFlip(),
      tr.Normalize_xception_tf(),
      tr.ToTensor_()])

    # voc_val = val(split='val', transform=composed_transforms_ts)
    # voc_val_f = val_flip(split='val', transform=composed_transforms_ts_flip)

    # testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=4)
    # testloader_flip = DataLoader(voc_val_f, batch_size=1, shuffle=False, num_workers=4)

    testloader_list.append(composed_transforms_ts)
    testloader_flip_list.append(composed_transforms_ts_flip)

  print("Eval Network")

  start_time = timeit.default_timer()
  # One testing epoch
  total_iou = 0.0

  c1, c2, p1, p2, a1, a2 = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],\
               [[0], [1, 2, 4, 13], [5, 6, 7, 10, 11, 12], [3, 14, 15], [8, 9, 16, 17, 18, 19]], \
               [[0], [1, 2, 3, 4, 5, 6]], [[0], [1], [2], [3, 4], [5, 6]], [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],\
               [[0], [1, 2, 3, 11], [4, 5, 7, 8, 16, 17], [14, 15], [6, 9, 10, 12, 13]]

  net.set_category_list(c1, c2, p1, p2, a1, a2)

  net.eval()


  # load image 
  if 1:
    if 0:
      im_name = 'demo.jpg'
      dir_frames = f'/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Self-Correction-Human-Parsing/mhp_extension/data/DemoDataset/global_pic/'
      list_frame= [im_name]
      
    if 0:
      dir_name = 'freeweights'

      # One-Arm_Dumbbell_Row/4IoyUvtF7do/3.000_39.188
      class_name = 'One-Arm_Dumbbell_Row'
      vid = '4IoyUvtF7do'
      clip = '3.000_39.188'
      dir_fw = '/nethome/hkwon64/Research/imuTube/dataset/imutube_v2'
      dir_frames = dir_fw + f'/{dir_name}/{class_name}/{vid}/{clip}/frames'

      list_frame = os.listdir(dir_frames)
      list_frame = [item for item in list_frame if item[-4:] == '.png']
      list_frame.sort()

    if 0:
      dir_name = 'freeweights'

      # Incline_Dumbbell_Press/4UZ8G8eW5MU/17.000_36.726
      class_name = 'Incline_Dumbbell_Press'
      vid = '4UZ8G8eW5MU'
      clip = '17.000_36.726'

    if 1:
      dir_name = 'freeweights'

      # One-Arm_Dumbbell_Row/Hfxxc4zg5zs/138.544_231.846
      class_name = 'One-Arm_Dumbbell_Row'
      vid = 'Hfxxc4zg5zs'
      clip = '138.544_231.846'

    sample_info = f'{class_name}/{vid}/{clip}'

    dir_fw = '/nethome/hkwon64/Research/imuTube/dataset/imutube_v2'
    dir_clip = dir_fw + f'/{dir_name}/{sample_info}'

    dir_frames = dir_clip + f'/frames'

    list_frame = os.listdir(dir_frames)
    list_frame = [item for item in list_frame if item[-4:] == '.png']
    list_frame.sort()

    dir_pose2d = dir_clip + '/pose2D'
    file_ap = dir_pose2d +'/alphapose-results.json'

    ap_results = json.load(open(file_ap))
    print ('load from ...', file_ap)
    # pprint (ap_results)
    # assert False

    frame_info = {}
    for result in ap_results:
      t = int(os.path.splitext(result['image_id'])[0])
      if t not in frame_info:
        frame_info[t] = {
          'pIDs': [],
          'kps': [],
          'score': [],
          'bbox': [],
          'bbox_expand': []}
          
      # idx = result['idx']
      # if type(idx) is list:
      #   idx = idx[0]
      #   if type(idx) is list:
      #     idx = idx[0]
      # # if len (idx) > 1:
      # # 	pprint (result)
      # # 	print(len(result['keypoints']))
      # # 	assert False
      # # print (idx)
      # # assert False
      # frame_info[t]['idx'].append(idx)

      kps = np.array(result['keypoints']).reshape((-1, 3))
      frame_info[t]['kps'].append(kps)
      
      _p_score = result['score']
      frame_info[t]['score'].append(_p_score)

      # get maximal bbox
      start_point = np.amin(kps[:,:2], axis=0).astype(int)
      end_point = np.amax(kps[:,:2], axis=0).astype(int)

      x1, y1, w, h = result['box']
      if x1 < start_point[0]:
          start_point[0] = int(x1)
      if y1 < start_point[1]:
          start_point[1] = int(y1)
      if x1+w > end_point[0]:
          end_point[0] = int(x1+w)
      if y1+h > end_point[1]:
          end_point[1] = int(y1+h)
          
      x_min, y_min = start_point
      x_max, y_max = end_point
      bbox = np.array([x_min, y_min, x_max, y_max])
      frame_info[t]['bbox'].append(bbox)

      # # get expanded bbox
      # exp_x_min, exp_y_min, exp_x_max, exp_y_max = func_bbox_expand(cfg.video.h, cfg.video.w, bbox, exp_ratio)

      # if exp_x_min == 0 \
      # and exp_y_min == 0 \
      # and exp_x_max == cfg.video.w-1 \
      # and exp_y_max == cfg.video.h-1:
      #   # print (f'{dir_clip} [{t}] fills whole image')
      #   frame_info[t]['bbox_expand'].append(bbox)
      # else:
      #   frame_info[t]['bbox_expand'].append([exp_x_min, exp_y_min, exp_x_max, exp_y_max])      

    vis_dir = f'/pascal_{dir_name}_vis/'
    mat_dir = f'/pascal_{dir_name}/'

  else:
    # im_name = '2008_000003.jpg'
    # im_name = '2008_000008.jpg'
    # im_name = '2008_000026.jpg'
    im_name = '2008_000041.jpg'
    # im_name = '2008_000034.jpg'
    dir_frames = '/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Grapy-ML/data/datasets/pascal/JPEGImages/'
    list_frame= [im_name]

  if not os.path.exists(opts.output_path + vis_dir):
    os.makedirs(opts.output_path + vis_dir)
  if not os.path.exists(opts.output_path + mat_dir):
    os.makedirs(opts.output_path + mat_dir)

  exp_ratio = 1.2

  with torch.no_grad():

    for t, im_name in enumerate(list_frame):
      t = 279
      im_name = list_frame[t]
      file_input = dir_frames + f'/{im_name}'

      _img = Image.open(file_input).convert('RGB') # return is RGB pic
      w, h = _img.size

      pID = 2
      bbox = frame_info[t]['bbox'][pID]
      exp_x_min, exp_y_min, exp_x_max, exp_y_max = func_bbox_expand(h, w, bbox, exp_ratio)
      bbox_expand = [exp_x_min, exp_y_min, exp_x_max, exp_y_max]

      x_min, y_min, x_max, y_max = bbox_expand
      kps = frame_info[t]['kps'][pID]
      # kps[:,:2] -= np.array([[x_min, y_min]])

      sample1 = []
      for composed_transforms_ts in testloader_list:
        _img = Image.open(file_input).convert('RGB') # return is RGB pic d
        _img = _img.crop(bbox_expand)
        
        if 0:
          w, h = _img.size
          ow = int(w*0.5)
          oh = int(h*0.5)
          _img = _img.resize((ow, oh), Image.BILINEAR)
          # print (_img.size)
          # assert False

        _img = composed_transforms_ts({'image': _img})
        sample1.append(_img)
      
      sample2 = []
      for composed_transforms_ts_flip in testloader_flip_list:
        _img = Image.open(file_input).convert('RGB') # return is RGB pic
        _img = _img.crop(bbox_expand)

        if 0:
          w, h = _img.size
          ow = int(w*0.5)
          oh = int(h*0.5)
          _img = _img.resize((ow, oh), Image.BILINEAR)
          # print (_img.size)
          # assert False
        
        _img = composed_transforms_ts_flip({'image': _img})
        sample2.append(_img)

      # print(ii)
      #1 0.5 0.75 1.25 1.5 1.75 ; flip:
      # sample1 = large_sample_batched[:6]
      # sample2 = large_sample_batched[6:]

      # for iii,sample_batched in enumerate(zip(sample1,sample2)):
      # 	print (sample_batched[0]['image'].shape)
      # 	print (sample_batched[1]['image'].shape)
      # assert False

      for iii,sample_batched in enumerate(zip(sample1,sample2)):
        # print (sample_batched[0]['image'].shape)
        # print (sample_batched[1]['image'].shape)

        inputs = sample_batched[0]['image']
        inputs_f = sample_batched[1]['image']
        inputs = torch.cat((inputs, inputs_f), dim=0)

        if iii == 0:
          _,_,h,w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
          if gpu_id >= 0:
            inputs = inputs.cuda()
          # outputs = net.forward(inputs)
          # pdb.set_trace()
          outputs, outputs_aux = net.forward((inputs, num_dataset_lbl), training=False)

          # print(outputs.shape, outputs_aux.shape)
          if opts.dataset == 'cihp':
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
          elif opts.dataset == 'pascal':
            outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
          else:
            outputs = (outputs[0] + flip(flip_atr(outputs[1]), dim=-1)) / 2

          outputs = outputs.unsqueeze(0)

          if iii>0:
            outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
            outputs_final = outputs_final + outputs
          else:
            outputs_final = outputs.clone()

      ################ plot pic
      predictions = torch.max(outputs_final, 1)[1]
      prob_predictions = torch.max(outputs_final,1)[0]
      results = predictions.cpu().numpy()
      # print (np.unique(results))
      # assert False

      prob_results = prob_predictions.cpu().numpy()
      vis_res = decode_labels(results)

      dir_im = opts.output_path + vis_dir + f'/{sample_info}'
      os.makedirs(dir_im, exist_ok=True)
      dir_mat = opts.output_path + mat_dir + f'/{sample_info}'
      os.makedirs(dir_im, exist_ok=True)
      parsing_im = Image.fromarray(vis_res[0])
      parsing_im.save(dir_im + f'/{im_name}.png')
      cv2.imwrite(dir_mat + f'{im_name}', results[0,:,:])
      print ('save in ...', dir_mat + f'{im_name}')

      # draw mask
      img = cv2.imread(file_input)
      print ('load from ...', file_input)
      # img = img[y_min:y_max, x_min:x_max]

      for c in range(1, len(classes)):
        mask = results[0,:,:] == c
        _mask = np.zeros(img.shape[:2], dtype=bool)
        _mask[y_min:y_max, x_min:x_max] = mask
        img = draw_mask(img, _mask, thickness=3, color=colors[c-1])
      
      img = draw_skeleton(img, kps)
      
      dir_mask = dir_im +'_mask'
      os.makedirs(dir_mask, exist_ok=True)

      file_mask = dir_mask + f'/{im_name}_mask.png'
      cv2.imwrite(file_mask, img)
      print ('save in ...', file_mask)		
      assert False

      # total_iou += utils.get_iou(predictions, labels)
    end_time = timeit.default_timer()
    print('time use for '+ f'{im_name}' + ' is :' + str(end_time - start_time))

  # Eval
  # pred_path = opts.output_path + mat_dir
  # eval_with_numpy(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file, dataset=opts.dataset)


if __name__ == '__main__':
  opts = get_parser()
  print (opts)
  # assert False

  main(opts)
  # pred_path = opts.output_path + '/atr_output/'
  # eval_with_numpy(pred_path=pred_path, gt_path=opts.gt_path,classes=opts.classes, txt_file=opts.txt_file, dataset=opts.dataset)
