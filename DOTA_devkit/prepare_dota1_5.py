import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain, DOTA2COCO_RotatedBBox
import argparse
# wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

wordname_16 = ['plane', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'soccer-ball-field', 'helicopter', 'container-crane']

def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default='/data/jifangcheng/datasets/dota')
    parser.add_argument('--dstpath', default=r'/data/jifangcheng/datasets/dota',
                        help='prepare data')
    args = parser.parse_args()

    return args

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)

def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')

def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    # if not os.path.exists(os.path.join(dstpath, 'test1024')):
    #     os.mkdir(os.path.join(dstpath, 'test1024'))
    new_train_name = 'train512_k'
    new_val_name = 'val512_k'

    if not os.path.exists(os.path.join(dstpath, new_train_name)):
        os.mkdir(os.path.join(dstpath, new_train_name))
    if not os.path.exists(os.path.join(dstpath, new_val_name)):
        os.mkdir(os.path.join(dstpath, new_val_name))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, new_train_name),
                      gap=20,
                      subsize=512,
                      num_process=32
                      )
    split_train.splitdata(1)

    # split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
    #                    os.path.join(dstpath, new_val_name),
    #                   gap=20,
    #                   subsize=512,
    #                   num_process=32
    #                   )
    # split_val.splitdata(1)

    # split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
    #                    os.path.join(dstpath, 'test1024', 'images'),
    #                   gap=200,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_test.splitdata(1)

    # DOTA2COCOTrain(os.path.join(dstpath, new_train_name), os.path.join(dstpath, new_train_name, 'train512.json'), wordname_16, difficult='-1')
    DOTA2COCO_RotatedBBox(os.path.join(dstpath, new_train_name), os.path.join(dstpath, new_train_name, 'train512.json'), wordname_16)
    # DOTA2COCO_RotatedBBox(os.path.join(dstpath, new_val_name), os.path.join(dstpath, new_val_name, 'val512.json'), wordname_16)
    # DOTA2COCOTest(os.path.join(dstpath, 'test1024'), os.path.join(dstpath, 'test1024', 'DOTA1_5_test1024.json'), wordname_16)

if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)
    # DOTA2COCOTrain(os.path.join(dstpath, 'trainval512'), os.path.join(dstpath, 'trainval512', 'DOTA1_5_trainval512.json'), wordname_16, difficult='-1')