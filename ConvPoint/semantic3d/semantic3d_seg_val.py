# Semantic3D Example with ConvPoint
import numpy as np
import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms

from sklearn.metrics import confusion_matrix, matthews_corrcoef

from PIL import Image

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('/content/perseverance/ConvPoint')
import utils.metrics as metrics

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)

# Part dataset only for training / validation
class PartDataset():

    def __init__(self, filelist, folder,
                    training=False, 
                    iteration_number=None,
                    block_size=8,
                    npoints=700,
                    nocolor=False):

        self.folder = folder
        self.training = training
        self.filelist = filelist
        self.bs = block_size
        self.nocolor = nocolor

        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False

        self.transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4)

    def __getitem__(self, index): 
        # print(index)       
        # load the data
        if self.training:
          index = random.randint(0, len(self.filelist)-1)  # index % len(self.filelist)  # 
        pts = np.load(os.path.join(self.folder, self.filelist[index]))
        # print(os.path.join(self.folder, self.filelist[index]))

        # get the features
        fts = pts[:,3:6]

        # get the labels
        lbs = pts[:, 6].astype(int)-1 # the generation script label starts at 1

        # get the point coordinates
        pts = pts[:, :3]

        """# pick a random point
        pt_id = random.randint(0, pts.shape[0]-1)
        pt = pts[pt_id]

        # create the mask
        mask_x = np.logical_and(pts[:,0]<pt[0]+self.bs/2, pts[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(pts[:,1]<pt[1]+self.bs/2, pts[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = pts[mask]
        lbs = lbs[mask]
        fts = fts[mask]"""
        
        # random selection
        if self.training:
          choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
          pts = pts[choice]
          lbs = lbs[choice]
          fts = fts[choice]

        # data augmentation
        if self.training:
            # random rotation
            pts = rotate_point_cloud_z(pts)

            # random jittering
            fts = fts.astype(np.uint8)
            fts = np.array(self.transform(Image.fromarray(np.expand_dims(fts, 0))))
            fts = np.squeeze(fts, 0)
        
        fts = fts.astype(np.float32)
        fts = fts / 255 - 0.5

        if self.nocolor:
            fts = np.ones((pts.shape[0], 1))

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return self.iterations

def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels, args=args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processeddir', type=str, default='./data/processed/')
    parser.add_argument("--savedir", type=str, default='./data/training_results/')
    parser.add_argument('--block_size', type=float, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--npoints", type=int, default=750)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--model", type=str, default="SegBig")
    parser.add_argument("--drop", type=float, default=0.5)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    training_folder = os.path.join(args.savedir, "{}_{}_nocolor{}_drop{}_lr{}_batch{}_epochs{}_iter{}_{}".format(
            args.model, args.npoints, args.nocolor, args.drop, args.lr,args.batch_size,args.epochs,args.iter, time_string))

    train_dir = args.processeddir + 'train/pointcloud/'
    val_dir = args.processeddir + 'val/pointcloud/'
    filelist_train = [f for f in os.listdir(train_dir)]
    filelist_val = [f for f in os.listdir(val_dir)]
	
    N_CLASSES = 5

    # create model
    print("Creating the network...", end="", flush=True)
    if args.nocolor:
        net = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    else:
        net = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)
    net.cuda()
    print("Done")

    ##### TRAIN
    print("Create the datasets...", end="", flush=True)
    ds_train = PartDataset(
        filelist_train,
        train_dir,
        training=True,
        iteration_number=args.batch_size*args.iter,
        npoints=args.npoints,
        nocolor=args.nocolor,
	)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    
    ds_val = PartDataset(
      filelist_val,
      val_dir,########
      training=False,
      iteration_number=len(filelist_val),
      nocolor=args.nocolor,
	)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=args.threads)
    
    print("Done")

    print("Create optimizer...", end="", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    print("Done")
    
    # create the root folder
    os.makedirs(training_folder, exist_ok=True)
    
    # create the log file
    logs = open(os.path.join(training_folder, "log.txt"), "w")
    # iterate over epochs
    # weight = torch.from_numpy(np.array([])).float()
    for epoch in range(args.epochs):
        ######## training
        net.train()

        actuals = np.array([], dtype=np.int)
        predictions = np.array([], dtype=np.int)
        cm_train = np.zeros((N_CLASSES, N_CLASSES))
        train_loss = 0
        t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
        for pts, features, seg in t:
            features = features.cuda()
            pts = pts.cuda()
            seg = seg.cuda()
            
            optimizer.zero_grad()
            outputs = net(features, pts)
            loss = F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy().ravel()
            target_np = seg.cpu().numpy().copy().ravel()
			
            actuals = np.concatenate((actuals, target_np))
            predictions = np.concatenate((predictions, output_np))
            cm_ = confusion_matrix(target_np, output_np, labels=list(range(N_CLASSES)))
            cm_train += cm_

            oa_train = f"{metrics.stats_overall_accuracy(cm_train):.4f}"
            iou_train = f"{metrics.stats_iou_per_class(cm_train)[0]:.4f}"
            mcc_train= f"{matthews_corrcoef(actuals, predictions):.4f}"
            train_loss += loss.detach().cpu().item()
			
            t.set_postfix(OA=wblue(oa_train), MCC=wblue(mcc_train), IOU=wblue(iou_train), LOSS=wblue(f"{train_loss/cm_train.sum():.4e}"))

        # save the checkpoints
        model_status_path = os.path.join(training_folder, 'val_state_dict.pth')
        torch.save(net.state_dict(), model_status_path)

        # validation
        actuals = np.array([], dtype=np.int)
        predictions = np.array([], dtype=np.int)
        net.eval()
        with torch.no_grad():
            for pts, features, seg in val_loader:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()
                
                outputs = net(features, pts)

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy().ravel()
                target_np = seg.cpu().numpy().copy().ravel()

                actuals = np.concatenate((actuals, target_np))
                predictions = np.concatenate((predictions, output_np))

        cm_val = confusion_matrix(target_np, output_np, labels=list(range(N_CLASSES)))
        oa_val = f"{metrics.stats_overall_accuracy(cm_val):.4f}"
        iou_val = f"{metrics.stats_iou_per_class(cm_val)[0]:.4f}"
        mcc_val = f"{matthews_corrcoef(actuals, predictions):.4f}"
        
        print(f"TRAIN: oa={oa_train} mcc={mcc_train} iou={iou_train} loss={train_loss/cm_train.sum():.4e}")#########
        print(f"VALID: oa={oa_val} mcc={mcc_val} iou={iou_val}")

        # write the logs
        logs.write(f"{epoch} {oa_train} {mcc_train} {iou_train} {train_loss/cm_train.sum():.4e} {oa_val} {mcc_val} {iou_val}\n")#######
        logs.flush()
    logs.close()

if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
