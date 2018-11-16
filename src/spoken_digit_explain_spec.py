# -*- coding: utf-8 -*-
'''
Script for explainable representation learning via
layerwise relevance propagation on spoken digit classification task.
'''
import os
import sys
import numpy as np
import json
import torch
import pdb

from PIL import Image

from models.AlexNet import AlexNet
from utils.datasets import prepare_data, SpokenDigits
from lrp.lrp import AlexNetLRP

# Parse arguments
if len(sys.argv) < 2:
    print('python {} configure.json'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    args = json.load(f)

# CUDA, randomness
if torch.cuda.is_available():
    print('CUDA available: True')
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.fastest = True
else:
    print('CUDA not available!')
    sys.exit(-1)
torch.manual_seed(args['random_seed'])
torch.cuda.manual_seed_all(args['random_seed'])

# Log
if not os.path.exists(args['img_save_dir']):
    os.makedirs(args['img_save_dir'])

# Data func
yield_train_test = prepare_data(args['data']['list'],
                                num_folds=args['num_folds'],
                                seed=args['random_seed'])

# Data
train_list, test_list = yield_train_test(9)  # last run
train_data = SpokenDigits(train_list, root=args['data']['path'])
test_data = SpokenDigits(test_list, root=args['data']['path'])
train_loader = torch.utils.data.DataLoader(
    train_data,
    args['batch_size'],
    shuffle=True,
    num_workers=args['num_workers']
    )
test_loader = torch.utils.data.DataLoader(
    test_data,
    args['batch_size'],
    shuffle=True,
    num_workers=args['num_workers']
    )

# Model
# AlexNet model
model = AlexNet(num_classes=args['AlexNet']['num_classes'])\
    .to(args['device'])
# load pretrained model state
checkpoint = torch.load(os.path.join(args['saved_model_dir'],
                                     args['model_to_eval']))
pretrained_model_state = checkpoint['model']
model.load_state_dict(pretrained_model_state)
# freeze model
for param in model.parameters():
    param.requires_grad = False


# AlexNet LRP model
model_Alex = AlexNetLRP(model).to(args['device'])
model_Alex.eval()


def forward_hook(self, input, output):
    '''
    Forward hook method for retrieving intermediate results.
    '''
    self.X = input[0]
    self.Y = output


# Forward hook
for i in range(0, len(model_Alex.layers)):
    model_Alex.layers[i].register_forward_hook(forward_hook)

correct_Alex = 0
tot_samples = 0
batch_size = args['batch_size']
save_dir = args['img_save_dir']

with torch.no_grad():
    for idx, (input, label) in enumerate(train_loader):
        tot_samples += label.size(0)

        input = input.to(args['device'])
        label = label.to(args['device'])

        output_Alex = model_Alex(input)
        pred_Alex = output_Alex.max(1, keepdim=True)[1]
        correct_Alex += pred_Alex.eq(label.view_as(pred_Alex)).\
            cpu().sum().item()

        T_Alex = (pred_Alex == torch.arange(args['AlexNet']['num_classes'],
                  dtype=pred_Alex.dtype).to(args['device'])).float()

        LRP_Alex = model_Alex.relprop(output_Alex * T_Alex)

        for i in range(0, batch_size):
            # Save results which are classified correctly by AlexNet
            #  if (pred_Alex.squeeze().cpu().numpy()[i] ==
                    #  label.cpu().numpy()[i]):
            if 1:
                img = input[i].squeeze().cpu().numpy()
                img = 255 * (img - img.min()) / (img.max() - img.min())
                img = img.astype('uint8')
                Image.fromarray(img, 'L').save(
                    '{}/{:d}_input_{:d}.JPEG'.format(
                        save_dir, idx * batch_size + i,
                        label.cpu().numpy()[i]))

                heatmap_Alex = LRP_Alex[i].squeeze().cpu().numpy()
                heatmap_Alex = np.absolute(heatmap_Alex)
                heatmap_Alex = 255 * (heatmap_Alex - heatmap_Alex.min()) /\
                    (heatmap_Alex.max() - heatmap_Alex.min())
                heatmap_Alex = heatmap_Alex.astype('uint8')
                Image.fromarray(heatmap_Alex, 'L').save(
                    '{}/{:d}_LRP_Alex_{:d}.JPEG'.format(
                        save_dir, idx * batch_size + i,
                        pred_Alex.squeeze().cpu().numpy()[i]))

    acc = correct_Alex / float(tot_samples)
    print('Accuracy = {:.4f}'.format(acc))
