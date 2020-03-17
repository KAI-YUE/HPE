# Python Libraries
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
from src.test import HLo_test, PRe_test, Synth_test, Dexter_test
from utils.tools import freeze_layers, load_pretrained_weights

def main(mode=None, model_path=None):
    """
    mode selection
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

    if (mode < 2):

        # Initialize the HLoNet
        if mode == 0:
            model = HLoNet()
        elif mode == 1:
            model = PReNet()
            load_pretrained_weights(config.pretrained_model_dir, model)
            freeze_layers(model, [0,1,2,3,4,5,6,7,8,9,10,11])

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
        elif mode == 1:
            PRe_train(model, optimizer, device)

    elif (mode < 6):
        state_dict = torch.load(model_path[0], map_location=device)

        # load from the checkpoint
        if (mode < 4):

            # Initialize the model and optimizer
            model = HLoNet() if mode == 2 else PReNet()
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            
            # Load the model and optimizer from the stored state_dict
            model.load_state_dict(state_dict['model'], strict=False)
            optimizer.load_state_dict(state_dict['optimizer'])
            optimizer.lr = config.learning_rate

            epoch = state_dict['epoch']
            del state_dict
            # Freeze certain layers
            freeze_layers(model, [0,1,2,3,4,5,6,7,8,9,10,11])

            if mode == 2:
                HLo_train(model, optimizer, device, epoch)
            elif mode == 3:
                PRe_train(model, optimizer, device, epoch)       
        
        elif (mode < 6):
            model = HLoNet() if mode == 4 else PReNet()
            model.to(device)

            model.load_state_dict(state_dict['model'], strict=False)
            model.eval()

            if mode == 4: 
                HLo_test(model, config.test_output_dir, device=device, mode=1)
            elif mode == 5:
                PRe_test(model, config.test_output_dir, device=device)

    else:
        Hal_dict = torch.load(model_path[0], map_location=device)
        Jor_dict = torch.load(model_path[1], map_location=device)

        HLo = HLoNet()
        HLo.eval()
        HLo.load_state_dict(Hal_dict['model'], strict=False)
        HLo.to(device)

        PRe = PReNet()
        PRe.eval()
        PRe.load_state_dict(Jor_dict['model'], strict=False)
        PRe.to(device)

        if mode == 6 :
            Synth_test(HLo, PRe, config.test_dir, config.test_output_dir)
        elif mode == 7:
            Dexter_test(HLo, PRe, config.test_dir, config.test_output_dir)   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=0, type=int, help='mode to select')
    parser.add_argument('--model', nargs='+')
    args = parser.parse_args()

    main(args.mode, args.model)
        