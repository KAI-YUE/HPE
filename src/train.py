# Python Libraries
import os
import time
import logging
import numpy as np

# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My Libraries
from src.loadConfig import loadConfig, log_Level
from src.dataset import HLoDataset, PReDataset
from src.loss import HLoCriterion, PReCriterion
from utils.tools import save_sample, save_model
from utils.kinematics import forward_kinematics

def HLo_train(model, optimizer, device="cuda", epoch=-1):
    config = loadConfig()

    train_data = HLoDataset(config.train_dir)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, drop_last=True, shuffle=True)
    
    val_data = HLoDataset(config.val_dir)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)

    sample_iter = val_data.create_iterator()
    
    # Initialize the loss function
    L =  HLoCriterion()
    start = time.time()

    # Initialize the logger
    Level = log_Level[config.log_level]
    logger = logging.getLogger(__name__)
    logger.setLevel(Level)
    
    fh = logging.FileHandler(config.log_file)
    fh.setLevel(Level)
    sh = logging.StreamHandler()
    sh.setLevel(Level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    
    while (epoch < config.max_epoch):
        epoch += 1
        iteration = 0
        for _, data in enumerate(train_loader, 0):
            
            optimizer.zero_grad()
            
            image = data['img'].to(device)
            heatmap = data['hm'].to(device)

            # Get output and calculate loss
            output = model(image)
            loss = L(output, heatmap)

            # backward for generator
            loss.backward()
            optimizer.step()
            
            # update the log
            if (config.log_interval and iteration % config.log_interval == 0):
                logger.info("epoch {} iter {} loss {:.3f}".format(epoch, iteration, loss))
            
            iteration += 1

            # if (iteration > 5):
            #     break
        
        # save the model
        if (config.save_epoch and iteration % config.save_epoch == 0):
            save_model(os.path.join(config.model_dir, 'HLo_epoch{}.pth'.format(epoch)), model, optimizer, epoch)

        # sample and save the prediction
        if(config.sample_epoch and epoch % config.sample_epoch == 0):
            
            model.eval()
            with torch.no_grad():

                image = data['img'].to(device)
                heatmap = data['hm'].to(device)
                output = model(image)

                save_sample(image, heatmap, output, epoch)

        # validate the model
        if(config.val_epoch and epoch % config.val_epoch == 0):
            
            model.eval()
            logger.info("="*80)
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    image = data['img'].to(device)
                    heatmap = data['hm'].to(device)
                    output = model(image)

                    loss = L(output, heatmap)
                    logger.info("val loss {}".format(loss))

                    if (i > config.sample_size):
                        break
            
        print('-'*80 + '\n{:.2f} h has elapsed'.format((time.time()-start)/3600))
        
        model.train()
        # Clear the cache after a epoch
        if (device != torch.device('cpu')):
            torch.cuda.empty_cache() 


def PRe_train(model, optimizer, device="cuda", epoch=-1):
    config = loadConfig()

    train_data = PReDataset(config.train_dir)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, drop_last=True, shuffle=True)
    
    val_data = PReDataset(config.val_dir)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)

    sample_iter = val_data.create_iterator()
    
    # Initialize the loss function
    L =  PReCriterion()
    start = time.time()

    # Initialize the logger
    Level = log_Level[config.log_level]
    logger = logging.getLogger(__name__)
    logger.setLevel(Level)
    
    fh = logging.FileHandler(config.log_file)
    fh.setLevel(Level)
    sh = logging.StreamHandler()
    sh.setLevel(Level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    while (epoch < config.max_epoch):
        epoch += 1
        iteration = 0
        for _, data in enumerate(train_loader, 0):
            
            optimizer.zero_grad()
            
            image = data['img'].to(device)
            pos = data['pos']
            scale = data["scale"].to(device)
            theta_alpha = data["theta_alpha"].to(device)

            # Get output and calculate loss
            output = model(image)
            loss = L(output, scale, theta_alpha)

            # backward for PRe
            loss.backward()
            optimizer.step()

            pred_pos = np.zeros((pos.shape[0], 21, 3))
            for i in range(pos.shape[0]):
                # pred_pos[i] = forward_kinematics(output["theta"][i].cpu().detach().squeeze().numpy(), output["scale"][i].cpu().detach().squeeze().numpy())
                pred_pos[i] = forward_kinematics(output["theta"][i].cpu().detach().squeeze().numpy(), scale[i].detach().cpu().numpy())
                # pred_pos[i] = forward_kinematics(theta_alpha[i].detach().cpu().numpy(), output["scale"][i].cpu().detach().squeeze().numpy())

            pos_loss = 1/pred_pos.shape[0] * np.sum(np.sqrt(np.sum((pred_pos - pos.squeeze().numpy())**2, -1)))

            # update the log
            if (config.log_interval and iteration % config.log_interval == 0):
                logger.info("epoch {} iter {} error {:.3f} DH_loss {:.3f}".format(epoch, iteration, pos_loss, loss))
            iteration += 1

            # if (iteration > 5):
            #     break
        
        # save the model
        if (config.save_epoch and iteration % config.save_epoch == 0):
            save_model(os.path.join(config.model_dir, 'PRe_epoch{}.pth'.format(epoch)), model, optimizer, epoch)

        # validate the model
        if(config.val_epoch and epoch % config.val_epoch == 0):
            
            model.eval()
            logger.info("="*80)
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    
                    image = data['img'].to(device)
                    heatmap = data['hm'].to(device)
                    pos = data['pos'].to(device)

                    output = model(image)

                    loss = L(output["pos"].to(device), pos)
                    logger.info("val loss {:.2f}".format(loss))

                    if (i > config.sample_size):
                        break
            
        print('-'*80 + '\n{:.2f} h has elapsed'.format((time.time()-start)/3600))
        
        model.train()
        # Clear the cache after a epoch
        if (device != torch.device('cpu')):
            torch.cuda.empty_cache() 