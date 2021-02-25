import numpy as np
import argparse
import os
import semantic3D_utils.lib.python.semantic3D as Sem3D


parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, default='./data/raw/')
parser.add_argument("--savedir", type=str, default='./data/processed/')
parser.add_argument("--voxel", type=float, default=0.1)
parser.add_argument("--checkfiles", action="store_true")
args = parser.parse_args()


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
def wred(str):
    return bcolors.FAIL+str+bcolors.ENDC


train_dir = args.rootdir + 'TRAIN'
val_dir = args.rootdir + 'VAL'
test_dir = args.rootdir + 'TEST'
filelist_train = [f.split('.')[0] for f in os.listdir(train_dir)]
filelist_val = [f.split('.')[0] for f in os.listdir(val_dir)]
filelist_test = [f.split('.')[0] for f in os.listdir(test_dir)]

print("Creating train directories...", end="", flush=True)
savedir = os.path.join(args.savedir, "train", "pointcloud_txt")
os.makedirs(savedir, exist_ok=True)
savedir_numpy = os.path.join(args.savedir, "train", "pointcloud")
os.makedirs(savedir_numpy, exist_ok=True)
print("done")

print("Generating train files...")
for filename in filelist_train:
    print(wblue(filename))
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    if os.path.exists(os.path.join(train_dir, filename_txt)):
        if os.path.exists(os.path.join(train_dir, filename_labels)):
            
            #if checkfiles flag, do not compute points
            if args.checkfiles: 
                continue

            # load file and voxelize
            Sem3D.semantic3d_load_from_txt_voxel_labels(os.path.join(train_dir, filename_txt),
                                                        os.path.join(train_dir, filename_labels),
                                                        os.path.join(savedir, filename+"_voxels.txt"),
                                                        args.voxel
                                                        )
            
            # save the numpy data
            np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))
        else:
            print(wred(f'Error -- label file does not exists: {os.path.join(train_dir, filename_labels)}'))
    else:
        print(wred(f'Error -- points file does not exists: {os.path.join(train_dir, filename_txt)}'))
print("Done")

print("Creating val directories...", end="", flush=True)
savedir = os.path.join(args.savedir, "val", "pointcloud_txt")
os.makedirs(savedir, exist_ok=True)
savedir_numpy = os.path.join(args.savedir, "val", "pointcloud")
os.makedirs(savedir_numpy, exist_ok=True)
print("done")

print("Generating val files...")
for filename in filelist_val:
    print(wblue(filename))
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    if os.path.exists(os.path.join(val_dir, filename_txt)):
        if os.path.exists(os.path.join(val_dir, filename_labels)):
            
            #if checkfiles flag, do not compute points
            if args.checkfiles: 
                continue

            # load file and voxelize
            Sem3D.semantic3d_load_from_txt_voxel_labels(os.path.join(val_dir, filename_txt),
                                                        os.path.join(val_dir, filename_labels),
                                                        os.path.join(savedir, filename+"_voxels.txt"),
                                                        args.voxel
                                                        )
            
            # save the numpy data
            np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))
        else:
            print(wred(f'Error -- label file does not exists: {os.path.join(val_dir, filename_labels)}'))
    else:
        print(wred(f'Error -- points file does not exists: {os.path.join(val_dir, filename_txt)}'))
print("Done")

print("Creating test directories...", end="", flush=True)
savedir = os.path.join(args.savedir, "test", "pointcloud_txt")
os.makedirs(savedir, exist_ok=True)
savedir_numpy = os.path.join(args.savedir, "test", "pointcloud")
os.makedirs(savedir_numpy, exist_ok=True)
print("done")

print("Generating test files...")
for filename in filelist_test:
    print(wblue(filename))
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    if os.path.exists(os.path.join(test_dir, filename_txt)):
        #if os.path.exists(os.path.join(test_dir, filename_labels)):
            
            #if checkfiles flag, do not compute points
            if args.checkfiles: 
                continue

            # load file and voxelize
            Sem3D.semantic3d_load_from_txt_voxel_labels(os.path.join(test_dir, filename_txt),
                                                        os.path.join(test_dir, filename_labels),
                                                        os.path.join(savedir, filename+"_voxels.txt"),
                                                        args.voxel
                                                        )
            
            # save the numpy data
            np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))
            #else:
            #print(wred(f'Error -- label file does not exists: {os.path.join(test_dir, filename_labels)}'))
    else:
        print(wred(f'Error -- points file does not exists: {os.path.join(test_dir, filename_txt)}'))
print("Done")
