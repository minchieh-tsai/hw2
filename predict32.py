# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:10:33 2020

@author: opgg
"""


from FCN32 import FCN32
from util import printw, transform_val

import os
import torch
import numpy as np
from P2Dataset import P2Dataset
from viz_mask import cls_color
import scipy.misc
import argparse 
from torchvision import transforms


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', help='img_dir', type=str)
    parser.add_argument('-o', '--output_dir', help='output_dir', type=str)
    args = parser.parse_args()
    
    # print(args.img_dir)
    # print(args.output_dir)
    # print(32)
    
    
    
    #%% load model
    log_name = "FCN32"
    prediction = True
    
    path_log = os.path.join("./code/p2/log", log_name, 'fcn32_epoch_15')
    path_out = args.output_dir
    root_val = args.img_dir
    # path_out = "./hw2_data/p2_data/validation"
    # root_val = "./output"
    
    feature_extract = True
    num_classes = 7
    
    # hyperparameters
    batch_size = 4
    
    # Initialize the model for this run
    model_ft = FCN32(num_classes=num_classes, feature_extract=feature_extract)
    
    
    checkpoint = torch.load(path_log, map_location='cpu')
    model_ft.load_state_dict(checkpoint['state_dict'])
    model_ft.eval()
    
    
    
    #%% load data
    
    dataset_val = P2Dataset(root=root_val, transform=transform_val, prediction=prediction)
    dataLoader_val = torch.utils.data.DataLoader(dataset_val, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printw("Use device", device)
        
    model_ft = model_ft.to(device)
    
                
    #%% save
    cmap = cls_color
    with torch.no_grad():
        for i, (inputs, _, file_name) in enumerate(dataLoader_val):
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            pred = torch.argmax(outputs, dim=1).unsqueeze(1)
            pred_cpu = pred.squeeze(1).detach().cpu().numpy()
            
            
            # transfer index 2 color & save
            mask = pred.transpose(1, 2).transpose(2, 3)
            for i in range(mask.size(0)):
                mask_save = np.empty([512, 512, 3])
                npmask = pred_cpu[i,:,:]
                indexs = np.unique(npmask)
                for index in indexs:
                    mask_save[npmask==index] = cmap[index]
                
                path_mask = os.path.join(path_out, file_name[i].replace('sat.jpg', 'mask.png'))
                
                transforms.ToPILImage(mask_save).save(path_mask)

                # torch.from_numpy(mask_save).save(path_mask)
                # scipy.misc.imsave(path_mask, np.uint8(mask_save))
            
    print("")
    printw("Done." , "")
    
    