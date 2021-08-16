from setting import *
from dataset import *
import argparse
import torch
from efficientdet-pytorch.effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from efficientdet-pytorch.effdet.efficientdet import HeadNet
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=512, help='image size')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--output', type=str, default='effdet5-augmix', help='folder to save checkpoint')
args = parser.parse_args()

class TrainGlobalConfig:
    num_workers = 0
    batch_size = args.batch_size
    n_epochs = args.epochs # n_epochs = 40
    lr = args.lr

    folder = args.output

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )

def collate_fn(batch):
    return tuple(zip(*batch))

def run_training():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)

def get_net():
    config = get_efficientdet_config('tf_efficientdet_d3')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('../input/efficientdet/efficientdet_d3-b0ea2cbc.pth')
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = args.image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

net = get_net()

if __name__ == "__main__":
    run_training()