# Pytorch Libraries
import os
import pickle
import torchvision

res50 = torchvision.models.resnet50(pretrained=True)
res50_statedict = res50.state_dict()

model_dict = \
{
    "conv1_w":        res50_statedict["conv1.weight"],
    
    "bn1_w":          res50_statedict["bn1.weight"],
    "bn1_b":       res50_statedict["bn1.bias"],
####################################################################
    
    "res2a_conv1_w":  res50_statedict["layer1.0.conv1.weight"],
    "res2a_conv2_w":  res50_statedict["layer1.0.conv2.weight"],
    "res2a_conv3_w":  res50_statedict["layer1.0.conv3.weight"],
    "res2a_conv4_w":  res50_statedict["layer1.0.downsample.0.weight"],
    
    "res2a_bn1_w":    res50_statedict["layer1.0.bn1.weight"],
    "res2a_bn1_b":    res50_statedict["layer1.0.bn1.bias"],
    "res2a_bn2_w":    res50_statedict["layer1.0.bn2.weight"],
    "res2a_bn2_b":    res50_statedict["layer1.0.bn2.bias"],
    "res2a_bn3_w":    res50_statedict["layer1.0.bn3.weight"],
    "res2a_bn3_b":    res50_statedict["layer1.0.bn3.bias"],
    "res2a_bn4_w":    res50_statedict["layer1.0.downsample.1.weight"],
    "res2a_bn4_b":    res50_statedict["layer1.0.downsample.1.bias"],      
    
    "res2b_conv1_w":  res50_statedict["layer1.1.conv1.weight"],
    "res2b_conv2_w":  res50_statedict["layer1.1.conv2.weight"],
    "res2b_conv3_w":  res50_statedict["layer1.1.conv3.weight"],
    
    "res2b_bn1_w":    res50_statedict["layer1.1.bn1.weight"],
    "res2b_bn1_b":    res50_statedict["layer1.1.bn1.bias"],
    "res2b_bn2_w":    res50_statedict["layer1.1.bn2.weight"],
    "res2b_bn2_b":    res50_statedict["layer1.1.bn2.bias"],
    "res2b_bn3_w":    res50_statedict["layer1.1.bn3.weight"],
    "res2b_bn3_b":    res50_statedict["layer1.1.bn3.bias"],

    "res2c_conv1_w":  res50_statedict["layer1.2.conv1.weight"],
    "res2c_conv2_w":  res50_statedict["layer1.2.conv2.weight"],
    "res2c_conv3_w":  res50_statedict["layer1.2.conv3.weight"],
    
    "res2c_bn1_w":    res50_statedict["layer1.2.bn1.weight"],
    "res2c_bn1_b":    res50_statedict["layer1.2.bn1.bias"],
    "res2c_bn2_w":    res50_statedict["layer1.2.bn2.weight"],
    "res2c_bn2_b":    res50_statedict["layer1.2.bn2.bias"],
    "res2c_bn3_w":    res50_statedict["layer1.2.bn3.weight"],
    "res2c_bn3_b":    res50_statedict["layer1.2.bn3.bias"],
############################################################################
    
    "res3a_conv1_w":  res50_statedict["layer2.0.conv1.weight"],
    "res3a_conv2_w":  res50_statedict["layer2.0.conv2.weight"],
    "res3a_conv3_w":  res50_statedict["layer2.0.conv3.weight"],
    "res3a_conv4_w":  res50_statedict["layer2.0.downsample.0.weight"],
    
    "res3a_bn1_w":    res50_statedict["layer2.0.bn1.weight"],
    "res3a_bn1_b":    res50_statedict["layer2.0.bn1.bias"],
    "res3a_bn2_w":    res50_statedict["layer2.0.bn2.weight"],
    "res3a_bn2_b":    res50_statedict["layer2.0.bn2.bias"],
    "res3a_bn3_w":    res50_statedict["layer2.0.bn3.weight"],
    "res3a_bn3_b":    res50_statedict["layer2.0.bn3.bias"],
    "res3a_bn4_w":    res50_statedict["layer2.0.downsample.1.weight"],
    "res3a_bn4_b":    res50_statedict["layer2.0.downsample.1.bias"],  

    "res3b_conv1_w":  res50_statedict["layer2.1.conv1.weight"],
    "res3b_conv2_w":  res50_statedict["layer2.1.conv2.weight"],
    "res3b_conv3_w":  res50_statedict["layer2.1.conv3.weight"],
    
    "res3b_bn1_w":    res50_statedict["layer2.1.bn1.weight"],
    "res3b_bn1_b":    res50_statedict["layer2.1.bn1.bias"],
    "res3b_bn2_w":    res50_statedict["layer2.1.bn2.weight"],
    "res3b_bn2_b":    res50_statedict["layer2.1.bn2.bias"],
    "res3b_bn3_w":    res50_statedict["layer2.1.bn3.weight"],
    "res3b_bn3_b":    res50_statedict["layer2.1.bn3.bias"],
    
    "res3c_conv1_w":  res50_statedict["layer2.2.conv1.weight"],
    "res3c_conv2_w":  res50_statedict["layer2.2.conv2.weight"],
    "res3c_conv3_w":  res50_statedict["layer2.2.conv3.weight"],
    
    "res3c_bn1_w":    res50_statedict["layer2.2.bn1.weight"],
    "res3c_bn1_b":    res50_statedict["layer2.2.bn1.bias"],
    "res3c_bn2_w":    res50_statedict["layer2.2.bn2.weight"],
    "res3c_bn2_b":    res50_statedict["layer2.2.bn2.bias"],
    "res3c_bn3_w":    res50_statedict["layer2.2.bn3.weight"],
    "res3c_bn3_b":    res50_statedict["layer2.2.bn3.bias"],
############################################################################
    
    "res4a_conv1_w":  res50_statedict["layer3.0.conv1.weight"],
    "res4a_conv2_w":  res50_statedict["layer3.0.conv2.weight"],
    "res4a_conv3_w":  res50_statedict["layer3.0.conv3.weight"],
    "res4a_conv4_w":  res50_statedict["layer3.0.downsample.0.weight"],
    
    "res4a_bn1_w":    res50_statedict["layer3.0.bn1.weight"],
    "res4a_bn1_b":    res50_statedict["layer3.0.bn1.bias"],
    "res4a_bn2_w":    res50_statedict["layer3.0.bn2.weight"],
    "res4a_bn2_b":    res50_statedict["layer3.0.bn2.bias"],
    "res4a_bn3_w":    res50_statedict["layer3.0.bn3.weight"],
    "res4a_bn3_b":    res50_statedict["layer3.0.bn3.bias"],
    "res4a_bn4_w":    res50_statedict["layer3.0.downsample.1.weight"],
    "res4a_bn4_b":    res50_statedict["layer3.0.downsample.1.bias"],  

    "res4b_conv1_w":  res50_statedict["layer3.1.conv1.weight"],
    "res4b_conv2_w":  res50_statedict["layer3.1.conv2.weight"],
    "res4b_conv3_w":  res50_statedict["layer3.1.conv3.weight"],
    
    "res4b_bn1_w":    res50_statedict["layer3.1.bn1.weight"],
    "res4b_bn1_b":    res50_statedict["layer3.1.bn1.bias"],
    "res4b_bn2_w":    res50_statedict["layer3.1.bn2.weight"],
    "res4b_bn2_b":    res50_statedict["layer3.1.bn2.bias"],
    "res4b_bn3_w":    res50_statedict["layer3.1.bn3.weight"],
    "res4b_bn3_b":    res50_statedict["layer3.1.bn3.bias"],
    
    "res4c_conv1_w":  res50_statedict["layer3.2.conv1.weight"],
    "res4c_conv2_w":  res50_statedict["layer3.2.conv2.weight"],
    "res4c_conv3_w":  res50_statedict["layer3.2.conv3.weight"],
    
    "res4c_bn1_w":    res50_statedict["layer3.2.bn1.weight"],
    "res4c_bn1_b":    res50_statedict["layer3.2.bn1.bias"],
    "res4c_bn2_w":    res50_statedict["layer3.2.bn2.weight"],
    "res4c_bn2_b":    res50_statedict["layer3.2.bn2.bias"],
    "res4c_bn3_w":    res50_statedict["layer3.2.bn3.weight"],
    "res4c_bn3_b":    res50_statedict["layer3.2.bn3.bias"],
###########################################################################
}

current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, "..\\checkpoints\\resnet50.dat"), "wb") as fp:
    pickle.dump(model_dict, fp)