import torch 
import numpy as np 
import os 
import cv2 

def torch_to_numpy(torch_image):
    np_image = (torch_image.cpu() * 255.0).numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    
    return np_image

def make_grid_image(np_image):
    '''
        np_image: N H W 3 
    '''
    n, H, W, c = np_image.shape
    grid_image = np.zeros((H*4, W*4, 3), np.uint8)

    for h in range(4):
        y_start = h*H
        y_end = y_start+H
        
        for w in range(4):
            x_start = w*W
            x_end = x_start+W
            
            idx = h*4 + w

            grid_image[y_start:y_end, x_start:x_end, :] = np_image[idx]


    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR) 
    return grid_image


def trainer(model, max_epoch, dataloader, loss_fn, optimizer, scheduler, logdir, device, test_every):

    for ep in range(1, max_epoch+1):
        train(model, ep, max_epoch, dataloader, loss_fn, optimizer, device)

        if scheduler:
            scheduler.step() 

        if ep % test_every == 0:
            test(model, ep, max_epoch, dataloader, logdir, device)

@torch.no_grad() 
def test(model, ep, max_epoch, dataloader, logdir, device):
    model.eval() 

    dst_root = os.path.join(logdir, str(ep)) 
    os.makedirs(dst_root, exist_ok=True)

    for i, elem in enumerate(dataloader):
        x = elem['x'].to(device)
        y = elem['y'].to(device)
        z = elem['z'].to(device)

        out = model(x, z)

        np_image = torch_to_numpy(y)
        grid_image = make_grid_image(np_image)
        dst_path = os.path.join(dst_root, 'y_{}.png'.format(i))
        cv2.imwrite(dst_path, grid_image)

        np_image = torch_to_numpy(out)
        grid_image = make_grid_image(np_image)
        dst_path = os.path.join(dst_root, 'pred_{}.png'.format(i))
        cv2.imwrite(dst_path, grid_image)
        


def train(model, ep, max_epoch, dataloader, loss_fn, optimizer, device):
    model.train() 

    running_loss = 0.0 
    for elem in dataloader:
        optimizer.zero_grad()

        x = elem['x'].to(device)
        y = elem['y'].to(device)
        z = elem['z'].to(device)

        out = model(x, z)

        loss = loss_fn(out, y)

        # update
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    print('[{}/{}] Loss: {:.4f}'.format(ep, max_epoch, running_loss))


