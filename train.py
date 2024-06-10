import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint ,
    save_checkpoint ,
    get_loaders ,
    check_accuracy , 
    save_predictions_as_imgs
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
# remember to change the "for TUTORIAL" line
TUTORIAL = True
TRAIN_IMG_DIR = "data/train_images/" if TUTORIAL else "/ssd3/CVPR2023comp/track1_train_data/seg/images/train/"
TRAIN_MASK_DIR = "data/train_masks/" if TUTORIAL else "/ssd3/CVPR2023comp/track1_train_data/seg/label/train/"
VAL_IMG_DIR = "data/val_images/" if TUTORIAL else "/ssd3/CVPR2023comp/track1_train_data/seg/images/val/"
VAL_MASK_DIR = "data/val_masks/" if TUTORIAL else "/ssd3/CVPR2023comp/track1_train_data/seg/label/val/"


# print(TRAIN_IMG_DIR)
# print(TRAIN_MASK_DIR)
# print(VAL_IMG_DIR)
# print(VAL_MASK_DIR)


def train(loader , model , optimizer , loss_fn , scaler , epoch) :
    loop = tqdm(loader)

    for batch_idx , (data , targets) in enumerate(loop) :

        data = data.to(device = DEVICE)
        
        targets = targets.float().unsqueeze(1).to(device = DEVICE) # for TUTORIAL
        # targets = targets.long().to(device = DEVICE)
        
        # data = data.type(torch.cuda.HalfTensor) # if Normalize is off

        # forward
        with torch.cuda.amp.autocast()  :
            predictions = model(data)
            loss = loss_fn(predictions , targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_description(("%10s") % (f"Epoch {epoch + 1}/{NUM_EPOCHS}"))
        loop.set_postfix(loss = "%.5f" % loss.item())


def main() :
    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT , width = IMAGE_WIDTH) ,
            A.Rotate(limit = 35 , p = 1.0) ,
            A.HorizontalFlip(p = 0.5) ,
            A.VerticalFlip(p = 0.1) ,
            A.Normalize(
                mean = [0.0 , 0.0 , 0.0] ,
                std = [1.0 , 1.0 , 1.0] ,
                max_pixel_value = 255.0 ,
            ) ,
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT , width = IMAGE_WIDTH) ,
            A.Normalize(
                mean = [0.0 , 0.0 , 0.0] ,
                std = [1.0 , 1.0 , 1.0] ,
                max_pixel_value = 255.0 ,
            ) ,
            ToTensorV2()
        ]
    )

    model = UNET(in_channels = 3 , out_channels = 1).to(DEVICE)
    # Use cross entropy loss for multiclass segmentation, also need to change the output channel
    loss_fn = nn.BCEWithLogitsLoss() if TUTORIAL else nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = optim.Adam(model.parameters() , lr = LEARNING_RATE)

    train_loader , val_loader = get_loaders(
        TRAIN_IMG_DIR ,
        TRAIN_MASK_DIR ,
        VAL_IMG_DIR ,
        VAL_MASK_DIR ,
        BATCH_SIZE ,
        train_transform ,
        val_transform ,
        NUM_WORKERS ,
        PIN_MEMORY
    )

    # n = 0
    # for index, batch in enumerate(train_loader): 
    #     if n >= 5 :
    #         break
    #     X, y = batch
    #     X, y = X.to(DEVICE), y.to(DEVICE)
    #     if n == 0 :
    #         print(type(X))
    #         print(X.shape)
    #         print(type(y))
    #         print(y.shape)
    #     n += 1

    # assert 1 == 2


    if LOAD_MODEL :
        load_checkpoint(torch.load("my_checkpoint.pth.tar"))

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS) :
        train(train_loader , model , optimizer , loss_fn , scaler , epoch)

        # save model
        checkpoint = {
            "state_dict" : model.state_dict() ,
            "optimizer" : optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        # print("Successfully Save!")

        # check accuracy
        check_accuracy(val_loader , model , device = DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader , model , folder = "saved_images/" , device = DEVICE
        )



if __name__ == "__main__" :
    main()
