﻿# Python Libraries
import os
import argparse

# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim

# My Libraries
from src.loadConfig import loadConfig
from src.networks import HLoNet, PReNet, init_weights
from src.train import HLo_train, PRe_train
from src.test import HLo_test, PRe_test

def main(mode=None, model_path=None):
    """
    mode to select
    0: train HLoNet from scratch 
    1: train PReNet from scratch
    2: train HLoNet from checkpoint
    3: train PReNet from checkpoint
    4: test HLoNet
    5: test PReNet
    6: test the joint model on synthHands dataset
    7: test the joint model on EgoDexter dataset
    """

    # Load configuration
    config = loadConfig()

    # Set the appropriate device property
    if (torch.cuda.is_available()):
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    if (mode == 0 or mode == 1):

        # Initialize the HLoNet
        model = HLoNet() if mode == 0 else PReNet()
        model.apply(init_weights)
        model.to(device)

        # Initialize the optimizer 
        optimizer = optim.Adam(
            params = model.parameters(),
            lr = config.learning_rate,
            weight_decay=0.0005
        )

        if mode == 0:
            HLo_train(model, optimizer, device)
        else:
            PRe_train(model, optimizer, device)

    elif (mode < 6):
        state_dict = torch.load(model_path[0], map_location=device)

        # load from the checkpoint
        if (mode == 2 or mode == 3):

            # Initialize the model and optimizer
            model = HLoNet() if mode == 2 else PReNet()
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            
            # Load the model and optimizer from the stored state_dict
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])

            if mode == 2:
                HLo_train(model, optimizer, device, state_dict['epoch'])
            else:
                PRe_train(model, optimizer, device, state_dict['epoch'])       
        
        elif (mode == 4 or 5):
            model = HLoNet() if mode == 4 else PReNet()
            model.to(device)

            model.load_state_dict(state_dict['model'])

            if mode == 4: 
                HLo_test(model, config.test_output_dir, device=device, mode=0)
            elif mode == 5:
                PRe_test(model, config.test_output_dir, device=device)

    else:
        Hal_dict = torch.load(model_path[0], map_location=device)
        Jor_dict = torch.load(model_path[1], map_location=device)

        HLo = HLoNet()
        HLo.load_state_dict(Hal_dict['model'])
        HLo.to(device)

        PRe = PReNet()
        PRe.load_state_dict(Jor_dict['model'])
        PRe.to(device)

        if mode == 6 :
            Synth_test(HLo, PRe, config.val_dir, config.test_output_dir)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=0, type=int, help='mode to select')
    parser.add_argument('--model', nargs='+')
    args = parser.parse_args()

    main(args.mode, args.model)
        