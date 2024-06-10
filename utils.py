import os
import torch
import torchvision
from dataset import Bdd100kDataset
from torch.utils.data import DataLoader
import numpy as np
from helpers.labels import trainId2label as t2l
from torchvision import transforms

def save_checkpoint(state , filename = "my_checkpoint.pth.tar") :
    print("=> Saving checkpoint")
    torch.save(state , filename)

def load_checkpoint(checkpoint , model) :
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir ,
    train_maskdir ,
    val_dir , 
    val_maskdir ,
    batch_size ,
    train_transform ,
    val_transform ,
    num_workers = 4 ,
    pin_memory = True
) :
    train_ds = Bdd100kDataset(
        image_dir = train_dir ,
        mask_dir = train_maskdir ,
        transform = train_transform
    )

    train_loader = DataLoader(
        train_ds ,
        batch_size = batch_size ,
        num_workers = num_workers ,
        pin_memory = pin_memory ,
        shuffle = True
    )

    val_ds = Bdd100kDataset(
        image_dir = val_dir ,
        mask_dir = val_maskdir ,
        transform = val_transform
    )

    val_loader = DataLoader(
        val_ds ,
        batch_size = batch_size ,
        num_workers = num_workers ,
        pin_memory = pin_memory ,
        shuffle = False
    )

    return train_loader , val_loader

def check_accuracy(loader , model , device = "cuda") :
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad() :
        for x , y in loader :
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            # for multiclass
            dice_score += (2 * (preds * y).sum()) / (preds +y).sum() + 1e-8

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )

    print(
        f"Dice score: {dice_score/len(loader):.4f}"
    )

    print()

    model.train()

def save_predictions_as_imgs(
    loader , model , folder = "saved_images/" , device = "cuda"
) :
    if not os.path.isdir(folder) :
        os.makedirs(folder)
    
    model.eval()
    for idx , (x , y) in enumerate(loader) :
        x = x.to(device = device)
        with torch.no_grad() :
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        
        # if preds.shape[0] == 3 :
        #     preds = (np.transpose(preds , (1 , 2 , 0)) + 1) / 2.0 * 255.0
        # elif preds.shape[0] == 1 :
        #     preds = (preds[0] + 1) / 2.0 * 255.0
        # print(preds.shape)

        # try to make prediction image
        # print(preds[0].shape)
        preds_to_image = torch.nn.functional.softmax(torch.unsqueeze(preds[0] , dim = 0) , dim = 1)
        # print("after softmax : " , preds_to_image.shape)
        pred_labels = torch.argmax(preds_to_image , dim = 1)
        # print("after argmax : " , pred_labels.shape)
        pred_labels = pred_labels.float()
        # print("after float : " , pred_labels.shape)

        # pred_labels = pred_labels.to('cpu')
        # pred_labels.apply_(lambda x: t2l[x].id)
        # print("after apply : " , pred_labels.shape)
        # pred_labels = pred_labels.to(device)
        # print("after to device : " , pred_labels.shape)

        pred_labels = transforms.Resize((1280, 720))(pred_labels)             
        # print("after transform : " , pred_labels.shape)

        pred_labels = transforms.ToPILImage()(pred_labels.byte())
        
        filename = f"saved_images/mask_{idx}.png"
        pred_labels.save(filename)
        # print("#"*80)
        # print("after softmax : " , preds_to_image.shape)
        # print(preds_to_image)
        # print("after argmax : " , pred_labels.shape)
        # print(np.unique(pred_labels.cpu()))
        # print(pred_labels)
        # print("#"*80)

        # torchvision.utils.save_image(pred_labels , f"{folder}/pred_{idx}.png")
        # print(y[0].shape)
        gt = y[0].unsqueeze(0)
        # print(gt.shape)
        gt = transforms.Resize((1280, 720))(gt)
        # print(gt.shape)
        gt = transforms.ToPILImage()(gt.byte())
        
        gt.save(f"{folder}{idx}.png")
        
        # torchvision.utils.save_image(y.unsqueeze(1) , f"{folder}{idx}.png")
    
    model.train()