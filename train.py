import argparse
from dataloader import get_dataloader
import torch
from torchvision import transforms
from runner import trainer
from models import get_model
import os 

def opt():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--mile_stone', nargs="+", type=int)
    parser.add_argument('--mode', type=str, default='dapi')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--test_every', type=int, default=50)
    parser.add_argument('--root_dir', type=str, default='./ML_Stain')
    parser.add_argument('--logdir', type=str, default='./log')
    return parser.parse_args() 

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using GPU')
        
    else:
        device = torch.device("cpu")
        print('Cannot use GPU!') 
        exit(1)

    os.makedirs(args.logdir, exist_ok=True) 

    # transformer 
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # dataloader 
    dataloader = get_dataloader(
        root_dir=args.root_dir, 
        mode=args.mode,
        transforms=TRANSFORMS, 
        batch_size=args.batch_size
    )
    
    # model 
    model = get_model() 
    model = model.to(device)

    # loss function & optimizer 
    loss_fn = torch.nn.L1Loss()
    #loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

    if args.mile_stone:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.mile_stone, gamma=0.1)
    else:
        scheduler = None 

    trainer(
        model=model, 
        max_epoch=args.epochs, 
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer, 
        scheduler = scheduler,
        logdir = args.logdir, 
        device = device, 
        test_every = args.test_every
    )

if __name__ == '__main__':
    args = opt() 
    main(args)