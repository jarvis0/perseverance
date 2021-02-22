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
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

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

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    print(indices.shape)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

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
    parser.add_argument("--trainingdir", type=str, required=True, help='SegBig batch_size_lr')
    parser.add_argument('--block_size', type=float, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--npoints", type=int, default=700)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--test_step", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="SegBig")
    parser.add_argument("--drop", type=float, default=0.5)
    args = parser.parse_args()

    train_dir = args.processeddir + '/train/pointcloud/'
    val_dir = args.processeddir + '/val/pointcloud'
    test_dir = args.processeddir + '/test/pointcloud/'
    filelist_train = [f for f in os.listdir(train_dir)] + [f for f in os.listdir(val_dir)]
    filelist_test = [f for f in os.listdir(test_dir)]
	
    N_CLASSES = 3

    # create model
    print("Creating the network...", end="", flush=True)
    if args.nocolor:
        net = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    else:
        net = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)
    if args.test:
        net.load_state_dict(torch.load(os.path.join(args.trainingdir, final_model.pth)))
    net.cuda()
    print("Done")


    ##### TRAIN
    if not args.test:
        print("Create the datasets...", end="", flush=True)
        ds = PartDataset(
          filelist_train,
          train_dir,
          training=True,
          block_size=args.block_size,
          iteration_number=args.batch_size*args.iter,
          npoints=args.npoints,
          nocolor=args.nocolor
		)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
        print("Done")

        print("Create optimizer...", end="", flush=True)
        optimizer = torch.optim.Adam(net.parameters(), args.lr)
        print("Done")
        
        # create the root folder
        os.makedirs(args.trainingdir, exist_ok=True)
        
        # create the log file
        logs = open(os.path.join(args.trainingdir, "final_log.txt"), "w")

        # iterate over epochs
        # weight = torch.from_numpy(np.array([])).float()
        for epoch in range(args.epochs):
            ######## training
            net.train()

            actuals = np.array([], dtype=np.int)
            predictions = np.array([], dtype=np.int)
            cm = np.zeros((N_CLASSES, N_CLASSES))
            train_loss = 0
            t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
            for pts, features, seg in t:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()
                
                optimizer.zero_grad()
                outputs = net(features, pts)
                loss = F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1), weight=None)
                loss.backward()
                optimizer.step()

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy().ravel()
                target_np = seg.cpu().numpy().copy().ravel()
				
                actuals = np.concatenate((actuals, target_np))
                predictions = np.concatenate((predictions, output_np))
                cm_ = confusion_matrix(target_np, output_np, labels=list(range(N_CLASSES)))
                cm += cm_

                oa = f"{metrics.stats_overall_accuracy(cm):.4f}"
                iou = f"{metrics.stats_iou_per_class(cm)[0]:.4f}"
                mcc = f"{matthews_corrcoef(actuals, predictions):.4f}"

                train_loss += loss.detach().cpu().item()
				
                t.set_postfix(OA=wblue(oa), MCC=wblue(mcc), IOU=wblue(iou), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))

            # save the checkpoints
            model_status_path = os.path.join(args.trainingdir, 'final_model.pth')
            torch.save(net.state_dict(), model_status_path)

            # write the logs
            logs.write(f"{epoch} {oa} {mcc} {iou} {train_loss/cm.sum():.4e}\n")
            logs.flush()

        logs.close()

    ##### TEST
    else:        
        print("Create the datasets...", end="", flush=True)
        ds_test = PartDataset(
            filelist_test,
            train_dir,
            training=False,
            iteration_number=len(filelist_test),
            nocolor=args.nocolor,
	    )
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=1)
        print("Done")

        logs = open(os.path.join(args.trainingdir, "final_log.txt"), "w")

        os.makedirs(os.path.join(args.trainingdir, "testing_results"), exist_ok=True)
        actuals = np.array([], dtype=np.int)
        predictions = np.array([], dtype=np.int)
        net.eval()
        with torch.no_grad():
            n = 0
            for pts, features, seg in test_loader:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()
                
                outputs = net(features, pts)

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy().ravel()
                target_np = seg.cpu().numpy().copy().ravel()

                actuals = np.concatenate((actuals, target_np))
                predictions = np.concatenate((predictions, output_np))

                save_fname = os.path.join(args.trainingdir, "testing_results", filelist_test[n])
                np.savetxt(save_fname, output_np, fmt='%d')
                n += 1

        cm_test = confusion_matrix(actuals, predictions, labels=list(range(N_CLASSES)))
        oa_test = f"{metrics.stats_overall_accuracy(cm_test):.4f}"
        iou_test = f"{metrics.stats_iou_per_class(cm_test)[0]:.4f}"
        mcc_test = f"{matthews_corrcoef(actuals, predictions):.4f}"
        
        print(f"TEST: oa={oa_test} mcc={iou_test} iou={mcc_test}")
    
    # write the logs
    logs.write(f"TEST {oa_test} {iou_test} {mcc_test}\n")
    logs.flush()
    logs.close()

    





        for filename in filelist_test:
            print(filename)
            ds = PartDatasetTest(
				filename,
				test_dir,
				block_size=args.block_size,
				npoints=args.npoints,
				test_step=args.test_step,
				nocolor=args.nocolor,
			)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

            xyzrgb = ds.xyzrgb[:,:3]
            scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
            with torch.no_grad():
                t = tqdm(loader, ncols=80)
                for pts, features, indices in t:
                    features = features.cuda()
                    pts = pts.cuda()
                    outputs = net(features, pts)

                    outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                    scores[indices.cpu().numpy().ravel()] += outputs_np
            
            mask = np.logical_not(scores.sum(1)==0)
            scores = scores[mask]
            pts_src = xyzrgb[mask]

            # create the scores for all points
            scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)

            # compute softmax
            scores = scores - scores.max(axis=1)[:,None]
            scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
            scores = np.nan_to_num(scores)

            os.makedirs(os.path.join(args.trainingdir, "testing_results"), exist_ok=True)

            # saving labels
            save_fname = os.path.join(args.trainingdir, "testing_results", filename)

            scores = scores.argmax(1)
            np.savetxt(save_fname,scores,fmt='%d')

            if args.savepts:
                save_fname = os.path.join(args.trainingdir, "testing_results", f"{filename}_pts.txt")

                xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
                np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])

            # break

if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
