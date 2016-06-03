# coding: utf-8
import os
import lasagne
import theano
import theano.tensor as T
import time
import itertools
import numpy as np
import glob
import h5py
import cPickle
import gzip
import matplotlib.pyplot as plt
import math
from PIL import Image
from PIL import ImageOps
from PIL import ImageStat
from PIL import ImageMath
from skimage.io import imread, imshow
from skimage.transform import resize, warp, rotate, rescale, SimilarityTransform
from skimage.util import pad, crop
from skimage import img_as_bool
import random
import thread
import copy
from operator import itemgetter
import cv2

dtimes = 16
# globle
BATCH_SIZE = 1
# inputNameList = []
# targetPath = []
Input_Shape = [(3,300,300)]
Output_Shape = 0
inputbatch1 = 0
inputbatch2 = 0
input_tmp = 0
isOK = False
CNum = "one"
targetbatch1 = 0
targetbatch2 = 0
target_tmp = 0

# two = False
# DataTrain = dict(mirror=True, flip=True, do_rotate=True, crop=True)
swpdict = dict(
    one="two",
    two="one"
)

def p(p):
    return random.random() <= p

def Data_getPicNameList(input_Dir, xFile_Type, tar_Dir=None, yFile_Type=None, Train_Val_Test_num=(0, 0, 1), Target_Mode='replace', repstr=[[0,'.png']]):
    inputNameList = []
    targetNameList= []
    if input_Dir.__class__ == '1'.__class__:
        print 'need [] input dir'
        return -1
    if (Target_Mode == 'replace') & (repstr.__class__ != [].__class__) & (repstr[0].__class__ != [].__class__):
        print 'need [[]]'
        return -1
    # if sum(Train_Val_Test_num) !=1:
    #     print 'sum(Train_Val_Test_num) must equle to 0'
    #     return -1
    print("start getting the name list of images")
    for idx in range(len(input_Dir)):
        inputNameList.extend([[]])
        inputNameList[idx] = glob.glob(input_Dir[idx] + '/*' + xFile_Type[idx])
        inputNameList[idx].sort()
    #
    # rtDict['X_train'+str(idx+1)] = inputNameList[:][0:TRAIN_set_num]
    # rtDict['X_valid'+str(idx+1)] = inputNameList[:][TRAIN_set_num:TRAIN_set_num + VALID_set_num]
    # rtDict['X_test'+str(idx+1)] = inputNameList[:][TRAIN_set_num + VALID_set_num::]
    if Target_Mode == 'replace' and tar_Dir == None:
        for idx in range(len(repstr)):
            targetNameList.extend([[]])
            targetNameList[idx] = [s.replace(xFile_Type[repstr[idx][0]], repstr[idx][1]) for s in inputNameList[idx]]
    elif tar_Dir:
        for idx in range(len(tar_Dir)):
            targetNameList.extend([[]])
            targetNameList[idx] = glob.glob(tar_Dir[idx] + '*' + yFile_Type[idx])
            targetNameList[idx].sort()
    try:
        print("get "+str(len(inputNameList[0]))+" images, " + str(len(targetNameList[0])) + " labels")
    except:
        print("get "+str(len(inputNameList[0]))+" images")

    datanum = len(inputNameList[0])
    TRAIN_set_num = int(Train_Val_Test_num[0] * datanum)
    VALID_set_num = int(Train_Val_Test_num[1] * datanum)
    TEST_set_num = int(Train_Val_Test_num[2] * datanum)

    l = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    return dict(
        X_train=inputNameList[:][0:TRAIN_set_num]
        if (inputNameList[:][0:TRAIN_set_num]!=[]) else l[0:len(inputNameList)],
        y_train=targetNameList[:][0:TRAIN_set_num]
        if (targetNameList[:][0:TRAIN_set_num]!=[]) else l[0:len(targetNameList)],
        X_valid=inputNameList[:][TRAIN_set_num:TRAIN_set_num + VALID_set_num]
        if (inputNameList[:][TRAIN_set_num:TRAIN_set_num + VALID_set_num] !=[]) else l[0:len(inputNameList)],
        y_valid=targetNameList[:][TRAIN_set_num:TRAIN_set_num + VALID_set_num]
        if (targetNameList[:][TRAIN_set_num:TRAIN_set_num + VALID_set_num]!=[]) else l[0:len(targetNameList)],
        X_test=inputNameList[:][TRAIN_set_num + VALID_set_num::]
        if (inputNameList[:][TRAIN_set_num + VALID_set_num::]!=[]) else l[0:len(inputNameList)],
        y_test=targetNameList[:][TRAIN_set_num + VALID_set_num::]
        if (targetNameList[:][TRAIN_set_num + VALID_set_num::]!=[]) else l[0:len(targetNameList)],
        )

def Data_readPics_thread(inputNamebatch, targetNamebatch, batchsize, genSetting='one_resize', arg=False, warpMode=None):
    global inputbatch1
    global inputbatch2
    global targetbatch1
    global targetbatch2
    global input_tmp
    global target_tmp
    global CNum
    global isOK
    datadict = dict(
        one=(inputbatch1, targetbatch1),
        two=(inputbatch2, targetbatch2)
    )
    input_tmp, target_tmp = datadict[CNum]
    if BATCH_SIZE == 1:
        inputNamebatch = [inputNamebatch]
        targetNamebatch = [targetNamebatch]
    WarpMode = [[],[]]
    for idx in range(len(inputNamebatch)):
        if warpMode == None or warpMode[0][idx] == 'pad2rect':
            WarpMode[0].append(1)
        if warpMode != None and warpMode[0][idx] == 'fcn':
            WarpMode[0].append(2)
        if warpMode != None and warpMode[0][idx] == 'fcnmax':
            WarpMode[0].append(3)
    for idx in range(len(targetNamebatch)):
        if warpMode == None or warpMode[0][idx] == 'pad2rect':
            WarpMode[1].append(1)
        if warpMode != None and warpMode[1][idx] == 'fcn':
            WarpMode[1].append(2)
        if warpMode != None and warpMode[1][idx] == 'fcnmax':
            WarpMode[1].append(3)
    for i in range(batchsize):
        imgs = []
        labels = []
        # imgs_Input = []
        # label_Input = []
        # SS = []
        if arg:
            r_scale = (random.random() * 0.2 + 1)
            # r_scale = 1#**************
            r_w = random.random()
            r_h = random.random()
            r_isrotate = p(0.3)
            r_rotate = r_isrotate * (random.randrange(-10, 10, 1) * 3.1415926 / 180)
            r_mirror = p(0.5)
            r_flip = p(0.5)
            # r_isAffine = p(0.3)
            # r_Afh = r_isAffine * random.random() * 0.1
            # r_Afv = r_isAffine * random.random() * 0.1
        for inpidx in range(len(inputNamebatch)):
            # print i
            img = Image.open(inputNamebatch[inpidx][i])
            ss = img.size
            if WarpMode[0][inpidx] == 1:
                ms = (max(ss) + max(ss) % 2)
                img = img.crop(((ss[0] - ms)/2, (ss[1] - ms)/2, (ss[0] - ms)/2 + ms, (ss[1] - ms)/2 + ms))
            if WarpMode[0][inpidx] == 2:
                Input_Shape[inpidx][2] = ss[0] + (dtimes - ss[0] % dtimes) % dtimes
                Input_Shape[inpidx][3] = ss[1] + (dtimes - ss[1] % dtimes) % dtimes
                img = img.crop((0, 0, Input_Shape[inpidx][2], Input_Shape[inpidx][3]))
                if i > 0:
                    assert('batchsize must be 1 while using mode fcn')
            if WarpMode[0][inpidx] == 3:
                rs = float(Input_Shape[inpidx][2]) / float(max(ss))
                ss = (int(float(ss[0]) * rs + 0.5), int(float(ss[1]) * rs + 0.5))
                img = img.resize(ss)
                ss = img.size
                Input_Shape[inpidx][2] = ss[0] + (dtimes - ss[0] % dtimes) % dtimes
                Input_Shape[inpidx][3] = ss[1] + (dtimes - ss[1] % dtimes) % dtimes
                img = img.crop((0, 0, Input_Shape[inpidx][2], Input_Shape[inpidx][3]))
                if i > 0:
                    assert('batchsize must be 1 while using mode fcn')
            imgs.append(img)
            if arg:
                size = int(Input_Shape[inpidx][2] * r_scale)
                img = img.resize((size, size))#176
                if r_isrotate:
                    img = img.rotate(r_rotate)
                if r_mirror:
                    img = ImageOps.mirror(img)
                if r_flip:
                    img = ImageOps.flip(img)
                dis = size - Input_Shape[inpidx][2]
                rh = int(dis * r_h)
                rw = int(dis * r_w)
                img = img.crop((rw, rh, rw + Input_Shape[inpidx][2], rh + Input_Shape[inpidx][3]))
            else:
                img = img.resize((Input_Shape[inpidx][2], Input_Shape[inpidx][3]))
                # img = img.crop((0, 0, Input_Shape[inpidx][2], Input_Shape[inpidx][3]))
            img.getdata()
            image = img.split()
            if len(image)==1:
                img = np.asarray((image[0]), dtype='float32')
                img = np.asarray((img - 103.93899999999999, img - 116.779, img - 123.68000000000001), dtype='float32')
            else:#BGR[103.93899999999999, 116.779, 123.68000000000001]
                img1 = np.asarray((image[2]), dtype='float32') - 103.93899999999999
                img2 = np.asarray((image[1]), dtype='float32') - 116.779
                img3 = np.asarray((image[0]), dtype='float32') - 123.68000000000001
                img = np.asarray((img1, img2, img3), dtype='float32')
            if WarpMode[0][inpidx] == 2:
                input_tmp[inpidx] = [img]
                # print(Input_Shape)
            elif WarpMode[0][inpidx] == 3:
                input_tmp[inpidx] = [img]
                Input_Shape[inpidx][2] = max(Input_Shape[inpidx])
                Input_Shape[inpidx][3] = max(Input_Shape[inpidx])
            else:
                input_tmp[inpidx][i] = img
        if genSetting == 'one_resize':
            for taridx in range(len(targetNamebatch)):
                label = Image.open(targetNamebatch[taridx][i]).convert("L")
                ss = label.size
                if WarpMode[1][taridx] == 1:
                    ms = (max(ss) + max(ss) % 2)
                    label = label.crop(((ss[0] - ms)/2, (ss[1] - ms)/2, (ss[0] - ms)/2 + ms, (ss[1] - ms)/2 + ms))
                    labels.append(label)
                    if arg:
                        size = int(ms * r_scale)
                        label = label.resize((size, size))#176
                        if r_isrotate:
                            label = label.rotate(r_rotate)
                        if r_mirror:
                            label = ImageOps.mirror(label)
                        if r_flip:
                            label = ImageOps.flip(label)
                        dis = size - ms
                        rh = int(dis * r_h)
                        rw = int(dis * r_w)
                        label = label.crop((rw, rh, rw + Input_Shape[inpidx][2], rh + Input_Shape[inpidx][3]))
                    else:
                        label = label.crop((0, 0, Input_Shape[inpidx][2], Input_Shape[inpidx][3]))
                if WarpMode[1][taridx] == 2:
                    Output_Shape[inpidx][2] = ss[0] + (dtimes - ss[0] % dtimes) % dtimes
                    Output_Shape[inpidx][3] = ss[1] + (dtimes - ss[1] % dtimes) % dtimes
                    if arg:
                        size = int(ms * r_scale)
                        label = label.resize((size, size))#176
                        if r_isrotate:
                            label = label.rotate(r_rotate)
                        if r_mirror:
                            label = ImageOps.mirror(label)
                        if r_flip:
                            label = ImageOps.flip(label)
                        dis = size - ms
                        rh = int(dis * r_h)
                        rw = int(dis * r_w)
                        label = label.crop((rw, rh, rw + ss[0], rh + ss[1]))
                if WarpMode[1][inpidx] == 3:
                    rs = float(Input_Shape[inpidx][2]) / float(max(ss))
                    ss = (int(float(ss[0]) * rs + 0.5), int(float(ss[1]) * rs + 0.5))
                    label = label.resize(ss)
                    ss = label.size
                    Output_Shape[inpidx][2] = ss[0] + (dtimes - ss[0] % dtimes) % dtimes
                    Output_Shape[inpidx][3] = ss[1] + (dtimes - ss[1] % dtimes) % dtimes
                    if arg:
                        size = int(ms * r_scale)
                        label = label.resize((size, size))#176
                        if r_isrotate:
                            label = label.rotate(r_rotate)
                        if r_mirror:
                            label = ImageOps.mirror(label)
                        if r_flip:
                            label = ImageOps.flip(label)
                        dis = size - ms
                        rh = int(dis * r_h)
                        rw = int(dis * r_w)
                        label = label.crop((rw, rh, rw + ss[0], rh + ss[1]))
                for outnums in range(len(Output_Shape)):
                    outlabel = label.resize((Output_Shape[outnums][2], Output_Shape[outnums][3]))
                    outlabel = np.asarray(outlabel, dtype='float32') / 255
                    if WarpMode[1][inpidx] == 2:
                        target_tmp[outnums] = [outlabel]
                    elif WarpMode[1][inpidx] == 3:
                        target_tmp[outnums] = [outlabel]
                        Output_Shape[inpidx][2] = max(Output_Shape[inpidx])
                        Output_Shape[inpidx][3] = max(Output_Shape[inpidx])
                    else:
                        target_tmp[outnums][i] = outlabel
    isOK = True
    CNum = swpdict[CNum]
    # return inputsbatch, targetsbatch

def Data_iterate_minibatches(inputs, targets, batchsize, arg=False, genSetting=None, shuffle=False, warpMode=None):
    # assert len(inputs[0]) == len(targets[0])
    if shuffle:
        rinputs = copy.deepcopy(inputs)
        rtargets = copy.deepcopy(targets)
        indices = np.random.permutation(len(inputs[0]))
        for i in range(len(inputs[0])):
            for idx in range(len(inputs)):
                rinputs[idx][i] = inputs[idx][indices[i]]
            for idx in range(len(targets)):
                rtargets[idx][i] = targets[idx][indices[i]]
        inputs = rinputs
        targets = rtargets
        # inputs[:] = inputs[indices]
        # targets[:] = targets[indices]

    init = True
    global input_tmp
    global target_tmp
    global isOK
    for start_idx in range(0, len(inputs[0]) - batchsize*2 + 1, batchsize):
        # if (isOK == False) and (two == False):
        #     inputsbatch, targetsbatch = read_pics(inputs[start_idx:start_idx + batchsize], targets[start_idx:start_idx + batchsize], batchsize, crop, mirror, flip, rotate)
        # else:
        while isOK == False:
            if init:
                sl = range(start_idx,start_idx + batchsize)
                thread.start_new_thread(Data_readPics_thread, ([itemgetter(*sl)(i) for i in inputs], [itemgetter(*sl)(i) for i in targets], batchsize, genSetting, arg, warpMode))
                init = False
                # inputsbatch, targetsbatch = read_pics(inputs[start_idx:start_idx + batchsize], targets[start_idx:start_idx + batchsize], batchsize, crop, mirror, flip, rotate)
            time.sleep(0.01)
        inputsbatch, targetsbatch = input_tmp, target_tmp
        isOK = False
        sl = range(start_idx  + batchsize,start_idx + 2 * batchsize)
        thread.start_new_thread(Data_readPics_thread, ([itemgetter(*sl)(i) for i in inputs], [itemgetter(*sl)(i) for i in targets], batchsize, genSetting, arg, warpMode))
        # yield itertools.chain(inputsbatch, targetsbatch)
        yield inputsbatch + targetsbatch
    while isOK == False:
        time.sleep(0.01)
    inputsbatch, targetsbatch = input_tmp, target_tmp
    isOK = False
    # yield itertools.chain(inputsbatch, targetsbatch)
    yield inputsbatch + targetsbatch
    # len(inputs) - batchsize*2 + 1

def Fun_Val_create(inputs, outputs):
    iter_valid = theano.function(inputs, outputs)
    return iter_valid

def Fun_test(Dpath, filetype, setting, X_batch, test_prediction,  warpMode=None,  writeimg=".tmp"):
    batch_size=1
    global Input_Shape
    global Output_Shape
    global BATCH_SIZE
    global inputbatch1
    global inputbatch2
    global targetbatch1
    global targetbatch2
    global input_tmp
    global target_tmp
    BATCH_SIZE = batch_size
    Input_Shape = setting['Input_Shape']
    Output_Shape = setting['Output_Shape']
    output_layer = setting['output_layer']
    saveParamName = setting['saveParamName']
    warpMode = None

    inputbatch1=[]
    inputbatch2=[]
    for inpidx in range(len(Input_Shape)):
        inputbatch1.append(np.zeros((BATCH_SIZE, Input_Shape[inpidx][1], Input_Shape[inpidx][2], Input_Shape[inpidx][3]), dtype='float32'))
        inputbatch2.append(np.zeros((BATCH_SIZE, Input_Shape[inpidx][1], Input_Shape[inpidx][2], Input_Shape[inpidx][3]), dtype='float32'))

    targetbatch1 = []
    targetbatch2 = []
    for inpidx in range(len(Output_Shape)):
        targetbatch1.append(np.zeros((BATCH_SIZE, Output_Shape[inpidx][1], Output_Shape[inpidx][2], Output_Shape[inpidx][3]), dtype='float32'))
        targetbatch2.append(np.zeros((BATCH_SIZE, Output_Shape[inpidx][1], Output_Shape[inpidx][2], Output_Shape[inpidx][3]), dtype='float32'))
    input_tmp = inputbatch1
    target_tmp = targetbatch1
    BATCH_SIZE = batch_size
    f = gzip.open(saveParamName, 'rb')
    params = cPickle.load(f)
    f.close()
    lasagne.layers.set_all_param_values(output_layer, params)
    feature_fn = Fun_Val_create(X_batch, test_prediction)

    dataset = Data_getPicNameList(Dpath, filetype, Target_Mode=None)
    print('start testing')
    # batch_test_accuracies = []
    save_list = range(0, len(dataset['X_test'][0]) - batch_size + 1, batch_size)


    batchnum=0
    charsize = len(writeimg[0])
    for batch in Data_iterate_minibatches(dataset['X_test'], dataset['y_test'], batch_size, shuffle=False, warpMode=warpMode):
        batch_ypred = feature_fn(batch[0])
        if writeimg != None:
            save_listbatch = dataset['X_test'][0][save_list[batchnum]: save_list[batchnum] + batch_size]
            batchnum += 1
            for i in range(len(batch_ypred[0])):
                Image.fromarray((batch_ypred[0][i][0]*255).astype(np.uint8)).convert("L").save(save_listbatch[i][0:-charsize] + '_' + writeimg[0] + ".bmp", "bmp")

def fitTest(setting, batch_size=BATCH_SIZE, datasetName=None, Test = False, plot=False, writeimg=".tmp"):
    global Input_Shape
    global Output_Shape
    global BATCH_SIZE
    # global LEARNING_RATE
    # global MOMENTUM
    global inputbatch1
    global inputbatch2
    global targetbatch1
    global targetbatch2
    global input_tmp
    global target_tmp
    BATCH_SIZE = batch_size
    # LEARNING_RATE = learning_rate
    # MOMENTUM = momentum
    Input_Shape = setting['Input_Shape']
    Output_Shape = setting['Output_Shape']
    num_epochs = setting['num_epochs']
    dataset = setting['dataset']
    output_layer = setting['output_layer']
    saveParamName = setting['saveParamName']
    # X_batch = setting['X_batch']
    # y_batch = setting['y_batch']
    # accuracy = setting['accuracy']
    # loss_train = setting['loss_train']
    # loss_test = setting['loss_test']
    # test_prediction = setting['test_prediction']
    genSetting = setting['genSetting']
    FunSetting = setting['FunSetting']
    try:
        warpMode = setting['warpMode']
    except:
        print('warpMode is None')
        warpMode = None
    # datasetName = setting['datasetName

    # iter_funcs=[]
    inputbatch1=[]
    inputbatch2=[]
    for inpidx in range(len(Input_Shape)):
        inputbatch1.append(np.zeros((BATCH_SIZE, Input_Shape[inpidx][1], Input_Shape[inpidx][2], Input_Shape[inpidx][3]), dtype='float32'))
        inputbatch2.append(np.zeros((BATCH_SIZE, Input_Shape[inpidx][1], Input_Shape[inpidx][2], Input_Shape[inpidx][3]), dtype='float32'))

    targetbatch1 = []
    targetbatch2 = []
    for inpidx in range(len(Output_Shape)):

        targetbatch1.append(np.zeros((BATCH_SIZE, Output_Shape[inpidx][1], Output_Shape[inpidx][2], Output_Shape[inpidx][3]), dtype='float32'))
        targetbatch2.append(np.zeros((BATCH_SIZE, Output_Shape[inpidx][1], Output_Shape[inpidx][2], Output_Shape[inpidx][3]), dtype='float32'))
    input_tmp = inputbatch1
    target_tmp = targetbatch1

    All_train_losses = []
    All_valid_losses = []
    All_valid_accuracies = []
    # if Test == False:
        # iter_funcs = 0
    iter_funcs = Fun_create(Funset_test=FunSetting['Funset_test'], Funset_test_only=FunSetting['Funset_test_only'])
    # print("Starting training...")
    # now = time.time()
    # bestV = 100
    # try:
    #     for epoch in Train_iter(iter_funcs, dataset, batch_size, genSetting=genSetting):
    #         print("Epoch {} of {} took {:.3f}s".format(
    #             epoch['number'], num_epochs, time.time() - now))
    #         now = time.time()
    #         print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
    #         print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
    #         print("  validation accuracy:\t\t{:.6f} %%".format(
    #             epoch['valid_accuracy'] * 100))
    #         All_train_losses.append(np.copy(epoch['batch_train_losses']))
    #         All_valid_losses.append(epoch['batch_valid_losses'])
    #         All_valid_accuracies.append(epoch['batch_valid_accuracies'])
    #         if epoch['valid_accuracy'] < bestV:
    #             bestV = epoch['valid_accuracy']
    #             if epoch['number'] % 5 ==0 & epoch['number'] > 100:
    #                 print ('saving....')
    #                 plotParam=All_train_losses, All_valid_losses, All_valid_accuracies
    #                 AllParam = lasagne.layers.get_all_param_values([output_layer]), plotParam
    #                 saveParam(AllParam, saveParamName)
    #         if epoch['number'] >= num_epochs:
    #             break
    #         if epoch['number'] % 1 == 0:#*********************************
    #             plotParam=All_train_losses, All_valid_losses, All_valid_accuracies
    #             AllParam = lasagne.layers.get_all_param_values([output_layer]), plotParam
    #             saveParam(AllParam, saveParamName, str(epoch['number']))
    #
    # except KeyboardInterrupt:
    #     pass
    # plotParam=All_train_losses, All_valid_losses, All_valid_accuracies
    # AllParam = lasagne.layers.get_all_param_values([output_layer]), plotParam
    # saveParam(AllParam, saveParamName)
    # else:
    f = gzip.open(saveParamName, 'rb')
    params, plotParam = cPickle.load(f)
    All_train_losses, All_valid_losses, All_valid_accuracies = plotParam
    f.close()
    if plot:
        N = All_train_losses.__len__()
        NN=All_train_losses[0].__len__()*All_train_losses.__len__()
        NNN=All_valid_losses[0].__len__()*All_valid_losses.__len__()
        train_losses = np.reshape(np.asarray(All_train_losses),[NN])
        train_l = np.reshape(np.mean(np.asarray(All_train_losses),1),[N])
        valid_losses = np.reshape(np.asarray(All_valid_losses),[NNN])
        valid_l = np.reshape(np.mean(np.asarray(All_valid_losses),1),[N])
        valid_accuracies = np.reshape(np.asarray(All_valid_accuracies),[NNN])
        valid_a = np.reshape(np.mean(np.asarray(All_valid_accuracies),1),[N])
        plt.figure(1)
        ax1 = plt.subplot(211)
        plt.plot(range(NN), train_losses)
        ax2 = plt.subplot(212)
        plt.plot(range(N), train_l)
        plt.figure(2)
        bx1 = plt.subplot(211)
        plt.plot(range(NNN), valid_losses)
        bx2 = plt.subplot(212)
        plt.plot(range(N), valid_l)
        plt.figure(3)
        cx1 = plt.subplot(211)
        plt.plot(range(NNN), valid_accuracies)
        cx2 = plt.subplot(212)
        plt.plot(range(N), valid_a)
        plt.show()
        # plt.figure(4)
        # plt.plot(range(N), train_l)
        # plt.plot(range(N), valid_l)
        # plt.legend()
    lasagne.layers.set_all_param_values(output_layer, params)
    print("Starting testing...")
    now = time.time()
    if datasetName == None:
        Test_dict = Test_iter(iter_funcs, dataset, batch_size=batch_size, writeimg=writeimg)
    else:
        Test_dict = Test_only_iter(iter_funcs, datasetName=datasetName, batch_size=batch_size, writeimg=writeimg, genSetting=genSetting, warpMode=warpMode)
        # Test_dict = test_all_only(iter_funcs, datasetName=datasetName, batch_size=batch_size)
    print("test took {:.4f}s".format(time.time() - now))
    print("  training loss:\t\t{:.6f}".format(Test_dict['train_loss']))
    print("  validation loss:\t\t{:.6f}".format(Test_dict['valid_loss']))
    print("  validation accuracy:\t\t{:.6f} %%".format(Test_dict['valid_accuracy'] * 100))
    print("  test loss:\t\t{:.6f}".format(Test_dict['test_loss']))
    print("  test accuracy:\t\t{:.6f} %%".format(Test_dict['test_accuracy'] * 100))

    # file = h5py.File('testResult', 'w')
    # Result=np.empty((TEST_set_num, 36*36),dtype='float32')
    # file.create_dataset('data_set', (TEST_set_num, 36*36), data=Result)
    # num_batches_test = dataset['num_examples_test'] // batch_size
    # for b in range(num_batches_test):
    #     batch_test_loss, batch_test_accuracy, Result[b*10:(b+1)*10], batch_y = iter_funcs['test'](b)
    # file.close()

    return All_train_losses, All_valid_losses, All_valid_accuracies

# def Fun_create(Funset_train, Funset_test=None, Funset_val=None):
#     return 0


