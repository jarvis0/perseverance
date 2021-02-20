import numpy as np
import semantic3D_utils.lib.python.semantic3D as sem3D
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testdir', type=str, default='./data/raw/TEST/')
parser.add_argument("--savedir", type=str, default='./data/results/')
parser.add_argument("--refdata", type=str, default='./data/processed/test/pointcloud_txt/')
parser.add_argument("--reflabel", type=str, required=True)
args = parser.parse_args()

filenames = [f.split('.')[0] for f in os.listdir(args.testdir) if os.path.isfile(os.path.join(args.testdir, f))]
os.makedirs(args.savedir, exist_ok=True)

for fname in filenames:
    print(fname)
    data_filename = os.path.join(args.testdir, fname+".txt")
    dest_filaname = os.path.join(args.savedir, fname+".labels")
    refdata_filename = os.path.join(args.refdata, fname+"_voxels.txt")
    reflabel_filename = os.path.join(args.reflabel, fname+"_voxels.npy")

    sem3D.project_labels_to_pc(dest_filaname, data_filename, refdata_filename, reflabel_filename)
