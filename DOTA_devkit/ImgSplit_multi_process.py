"""
-------------
This is the multi-process version
"""
import os
import codecs
import numpy as np
import math
from dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
import dota_utils as util
import copy
from multiprocessing import Pool
from functools import partial
import time
from itertools import chain

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))



class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext = '.png',
                 padding=True,
                 num_process=8
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.pool = Pool(num_process)
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
        # pdb.set_trace()
    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calc_intersection(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        inter_ratio = inter_area / poly1_area
        return inter_poly, inter_ratio 

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue
            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def get_poly4_from_poly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue
            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly, [pos, (pos + 1) % 5]

    def adjust_object(self, inter_poly, obj_poly, vis_flags):
        # calculate the 2 matched points
        obj2inter = {}
        for i, flag in enumerate(vis_flags):
            if flag == 1:
                error = []
                for j in range(len(inter_poly)):
                    error.append(pow(obj_poly[2 * i] - inter_poly[j][0], 2) + pow(obj_poly[2 * i + 1] - inter_poly[j][1], 2))
                min_idx = np.array(error).argsort()[0]
                obj2inter[i] = min_idx
        assert len(obj2inter) >= 2

        # select the 4 right points 
        inter_idxes = []
        if vis_flags[0] == 1 and vis_flags[3] == 1:
            inter_idxes.append(obj2inter[3] - 1)
            inter_idxes.append(obj2inter[3])
            inter_idxes.append(obj2inter[0])
            inter_idxes.append((obj2inter[0] + 1) % len(inter_poly))
        else:
            for i, flag in enumerate(vis_flags):
                if flag == 1 and vis_flags[i - 1] != 1:
                    inter_idxes.append(obj2inter[i] - 1)
                    inter_idxes.append(obj2inter[i])
                if flag == 1 and vis_flags[i - 1] == 1:
                    inter_idxes.append(obj2inter[i])
                    inter_idxes.append((obj2inter[i] + 1) % len(inter_poly))
        # sort the 4 points as the original order
        out_poly = list(chain(*[inter_poly[idx] for idx in inter_idxes]))
        adjusted_poly = choose_best_pointorder_fit_another(out_poly, obj_poly)
        return adjusted_poly

    def save_patche_points(self, resizeimg, objects, subimgname, left, up, right, down, filter_area=50):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        # mask_poly = []
        img_poly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gt_poly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])

                inter_poly, inter_ratio = self.calc_intersection(gt_poly, img_poly)
                # filter the small object
                if inter_poly.area <= filter_area or inter_ratio < 0.2:
                    continue
                
                vis_flags = []
                for i in range(4):
                    if obj['poly'][2 * i] >= left and obj['poly'][2 * i] < right and obj['poly'][2 * i + 1] >= up and obj['poly'][2 * i + 1] < down:
                        vis_flags.append(1)         # if the point is in the image, set flag 1
                    else:
                        vis_flags.append(0)         # if the point is not in the image set flag 0

                if sum(vis_flags) < 2:  # points < 2
                    continue
                elif sum(vis_flags) == 2:
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    out_poly = self.adjust_object(out_poly, obj['poly'], vis_flags)
                else:       # visible points >= 3
                    out_poly = obj['poly']

                poly_sub = self.polyorig2sub(left, up, out_poly)
                outline = ' '.join(list(map(str, poly_sub)))
                outline += ' ' + ' '.join(list(map(str, vis_flags)))

                if inter_ratio == 1:
                    # poly_sub = self.polyorig2sub(left, up, obj['poly'])
                    # outline = ' '.join(list(map(str, poly_sub)))
                    # outline += ' 1 1 1 1'       # 1 means the keypoint is in the true keypoint
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                else:   
                    # inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    # out_poly = list(inter_poly.exterior.coords)[0: -1]
                    # if len(out_poly) < 4:
                    #     continue

                    # out_poly2 = []
                    # vis_flags = []
                    # for i in range(len(out_poly)):
                    #     out_poly2.append(out_poly[i][0])
                    #     out_poly2.append(out_poly[i][1])
                    #     vis_flag = 0
                    #     for j in range(4):
                    #         error = pow(out_poly[i][0] - obj['poly'][2*j], 2) + \
                    #                 pow(out_poly[i][1] - obj['poly'][2 * j + 1], 2)
                    #         if error < 2.0:
                    #             vis_flag = 1
                    #             break
                    #     vis_flags.append(vis_flag)
                    
                    # if sum(vis_flags) < 2:  # points < 2
                    #     continue

                    # if len(out_poly) == 5:
                    #     out_poly2, merged_idxes = self.get_poly4_from_poly5(out_poly2)
                    # # elif (len(out_poly) > 5):
                    # #     continue
                    # if self.choosebestpoint:
                    #     out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    # poly_sub = self.polyorig2sub(left, up, out_poly2)

                    # for index, item in enumerate(poly_sub):
                    #     if (item <= 1):
                    #         poly_sub[index] = 1
                    #     elif (item >= self.subsize):
                    #         poly_sub[index] = self.subsize
                    # outline = ' '.join(list(map(str, poly_sub)))
                    # outline += ' ' + ' '.join(list(map(str, vis_flags)))

                    if (inter_ratio > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')

        self.saveimagepatches(resizeimg, subimgname, left, up)

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down, filter_area=50):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= filter_area):
                    continue
                inter_poly, half_iou = self.calc_intersection(gtpoly, imgpoly)

                # print('writing...')
                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif (half_iou > 0):
                #elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        try:
            img = cv2.imread(os.path.join(self.imagepath, name + extent))
            print('img name:', name)
        except:
            print('img name:', name)
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            #obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        width = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        # if (max(width, height) < self.subsize):
        #     return

        left, up = 0, 0
        while (left < width):
            if (left + self.subsize >= width):
                left = max(width - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, width - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                # self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                self.save_patche_points(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= width):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """

        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]

        worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
        #
        # for name in imagenames:
        #     self.SplitSingle(name, rate, self.ext)
        self.pool.map(worker, imagenames)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == '__main__':
    # example usage of ImgSplit
    # start = time.clock()
    # split = splitbase(r'/data/dj/dota/val',
    #                    r'/data/dj/dota/val_1024_debugmulti-process_refactor') # time cost 19s
    # # split.splitdata(1)
    # # split.splitdata(2)
    # split.splitdata(0.4)
    #
    # elapsed = (time.clock() - start)
    # print("Time used:", elapsed)

    # split = splitbase(r'/data/dota2/train',
    #                    r'/data/dota2/train1024',
    #                   gap=200,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_train = splitbase(r'/data0/data_dj/dota2/train',
    #                   r'/data0/data_dj/dota2/trainval1024_ms',
    #                     gap=200,
    #                     subsize=1024,
    #                     num_process=30)
    # split_train.splitdata(1.5)
    # split_train.splitdata(0.5)
    #
    # split_val = splitbase(r'/data0/data_dj/dota2/val',
    #                       r'/data0/data_dj/dota2/trainval1024_ms',
    #                       gap=200,
    #                       subsize=1024,
    #                       num_process=30)
    # split_val.splitdata(1.5)
    # split_val.splitdata(0.5)


    # split = splitbase(r'/home/dingjian/project/dota2/test-c1',
    #                   r'/home/dingjian/project/dota2/test-c1-1024',
    #                   gap=512,
    #                   subsize=1024,
    #                   num_process=16)
    # split.splitdata(1)


    # split_train = splitbase(r'/data/mmlab-dota1.5/train',
    #                   r'/data/mmlab-dota1.5/split-1024/trainval1024_ms',
    #                     gap=200,
    #                     subsize=1024,
    #                     num_process=40)
    # split_train.splitdata(1.5)
    # split_train.splitdata(0.5)
    #
    # split_val = splitbase(r'/data/mmlab-dota1.5/val',
    #                       r'/data/mmlab-dota1.5/split-1024/trainval1024_ms',
    #                       gap=200,
    #                       subsize=1024,
    #                       num_process=40)
    # split_val.splitdata(1.5)
    # split_val.splitdata(0.5)
    #
    # split_train_single = splitbase('/data/mmlab-dota1.5/train',
    #                                '/data/mmlab-dota1.5/split-1024/trainval1024',
    #                                gap=200,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_single.splitdata(1)
    #
    # split_val_single = splitbase('/data/mmlab-dota1.5/val',
    #                              '/data/mmlab-dota1.5/split-1024/trainval1024',
    #                              gap=200,
    #                              subsize=1024,
    #                              num_process=40)
    # split_val_single.splitdata(1)

    # dota-1.5 1024 split new
    # split_train_single = splitbase('/data/mmlab-dota1.5/train',
    #                                '/data/mmlab-dota1.5/split-1024_v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_single.splitdata(1)
    #
    # split_train_ms = splitbase('/data/mmlab-dota1.5/train',
    #                                '/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_ms.splitdata(0.5)
    # split_train_ms.splitdata(1.5)
    #
    # # val
    # split_val_single = splitbase('/data/mmlab-dota1.5/val',
    #                                '/data/mmlab-dota1.5/split-1024_v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_val_single.splitdata(1)
    #
    # split_val_ms = splitbase('/data/mmlab-dota1.5/val',
    #                            '/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_val_ms.splitdata(0.5)
    # split_val_ms.splitdata(1.5)
    #
    # # test
    # split_test_single = splitbase('/data/mmlab-dota1.5/test',
    #                                '/data/mmlab-dota1.5/split-1024_v2/test1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_test_single.splitdata(1)
    #
    # split_test_ms = splitbase('/data/mmlab-dota1.5/test',
    #                            '/data/mmlab-dota1.5/split-1024_v2/test1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_test_ms.splitdata(0.5)
    # split_test_ms.splitdata(1.5)

    # split_train_single = splitbase(r'/data/data_dj/dota2/train',
    #                                r'/data/data_dj/dota2/split-1024-v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_single.splitdata(1)
    #
    # split_train_ms = splitbase(r'/data/data_dj/dota2/train',
    #                            r'/data/data_dj/dota2/split-1024-v2/trainval1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_train_ms.splitdata(0.5)
    # split_train_ms.splitdata(1.5)
    #
    #
    # split_val_single = splitbase(r'/data/data_dj/dota2/val',
    #                                r'/data/data_dj/dota2/split-1024-v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_val_single.splitdata(1)
    #
    # split_val_ms = splitbase(r'/data/data_dj/dota2/val',
    #                            r'/data/data_dj/dota2/split-1024-v2/trainval1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_val_ms.splitdata(0.5)
    # split_val_ms.splitdata(1.5)

    split_test_single = splitbase(r'/home/dingjian/project/dota2/test-dev',
                                  r'/home/dingjian/workfs/dota2_v2/split-1024-v2/test-dev1024',
                                  gap=512,
                                  subsize=1024,
                                  num_process=16)
    split_test_single.splitdata(1)

    split_test_ms = splitbase(r'/home/dingjian/project/dota2/test-dev',
                                  r'/home/dingjian/workfs/dota2_v2/split-1024-v2/test-dev1024_ms',
                                  gap=512,
                                  subsize=1024,
                                  num_process=16)
    # split_test_ms.splitdata(1)
    split_test_ms.splitdata(0.5)
    split_test_ms.splitdata(1.5)
