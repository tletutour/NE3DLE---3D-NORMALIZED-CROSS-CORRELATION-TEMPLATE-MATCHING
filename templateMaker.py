#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import os
from multiprocessing import Process
from scipy.ndimage import gaussian_filter
segpath="/media/tom/TOSHIBA EXT/visceral/segmentations/CTce_ThAb/"
fullpath="/media/tom/TOSHIBA EXT/visceral/volumes/CTce_ThAb"
recep_path="/media/tom/TOSHIBA EXT"
segfiles = os.listdir( segpath )#list of all files in working dir
typicalfilename="10000100_1_CTce_ThAb_2_0.nii.gz"

classtolabel={
    1247:"trachea",
    1302:"right lung",
    1326:"left lung",
    170:"pancreas",
    187:"gallbladder",
    237:"urinary bladder",
    2473:"sternum",
    29193:"first lumbar vertebra",
    29662:"right kidney",
    29663 :"left kidney",
    30324:"right adrenal gland",
    30325:"left adrenal gland",
    32248:"right psoas major",
    32249:"left psoas major",
    40357:"muscle body of right rectus abdominis",
    40358:"muscle body of left rectus abdominis",
    480:"aorta",
    58:"liver",
    7578:"thyroid gland",
    86:"spleen"
}
patients=["100","104","105","106","108","109","110","111","112","113","127","128","130","131","132","133","134","135","136"]
#patients=["100","104","105","106","108"]
patients_to_redo=["131","132","133","134","135","136"]
#classes=["58","1302","1247","1326","170","187" ,"237","2473" ,"29193" ,"29662" ,"29663" ,"30324" ,"30325" ,"32248" ,"32249" ,"40357" ,"40358" ,"480"  ,"7578" ,"86"]
classes=["58","1302","29193"]
#classes=["1302","1247","1326","170"]
#classes=["58"]

def get_class(filename):
    arr=filename.split("_")
    return arr[4]
def get_patient(filename):
    return filename[5:8]

def bbox2_3D(img):
    """
    Method openly copied from
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return img[rmin: rmax, cmin: cmax, zmin: zmax]

def merge(nib_mask,nib_img):#merges two nib images and returns a NP array !!!
    mask_arr=nib_mask.get_fdata()
    img_arr=nib_img.get_fdata()
    return np.where(mask_arr>0,img_arr,mask_arr)


def broadcast_3d(arr1,arr2):
    xmax = max(arr1.shape[0], arr2.shape[0])
    ymax = max(arr1.shape[1], arr2.shape[1])
    zmax = max(arr1.shape[2], arr2.shape[2])
    # broadcasting to same shape
    arr1 = np.pad(arr1, ((0, xmax - arr1.shape[0]), (0, ymax - arr1.shape[1]), (0, zmax - arr1.shape[2])), 'constant')
    arr2 = np.pad(arr2, ((0, xmax - arr2.shape[0]), (0, ymax - arr2.shape[1]), (0, zmax - arr2.shape[2])), 'constant')
    return arr1 , arr2

def broadcast_2d(arr1,arr2):
    xmax = max(arr1.shape[0], arr2.shape[0])
    ymax = max(arr1.shape[1], arr2.shape[1])
    # broadcasting to same shape
    arr1 = np.pad(arr1, ((0, xmax - arr1.shape[0]), (0, ymax - arr1.shape[1])), 'constant')
    arr2 = np.pad(arr2, ((0, xmax - arr2.shape[0]), (0, ymax - arr2.shape[1])), 'constant')
    return arr1 , arr2

def add_two(arr1,arr2):
    xmax = max(arr1.shape[0], arr2.shape[0])
    ymax = max(arr1.shape[1], arr2.shape[1])
    zmax = max(arr1.shape[2], arr2.shape[2])
    #broadcasting to same shape
    arr1=np.pad(arr1,((0,xmax-arr1.shape[0]),(0,ymax-arr1.shape[1]),(0,zmax-arr1.shape[2])),'constant')
    arr2=np.pad(arr2, ((0, xmax - arr2.shape[0]), (0, ymax - arr2.shape[1]), (0, zmax - arr2.shape[2])), 'constant')
    return arr1+arr2


def make_templates(excluded_patient):
    try:
        os.mkdir(os.path.join('/media/tom/TOSHIBA EXT/PIR',excluded_patient))
    except Exception as e: print(e)

    valid_dirs=[]
    for directory in segfiles:
        if get_patient(directory)!=excluded_patient and get_patient(directory) in patients:
            valid_dirs.append(directory)
    for classe in classes:
        print("starting class",classe)
        sumofall=np.zeros((512,512,512))
        affine_sum=np.zeros((4,4))
        nb=0
        for directory in valid_dirs:
            if(get_class(directory)==classe):
                nb+=1
                print("\t patient"+get_patient(directory))
                vol_filename="10000"+get_patient(directory)+"_1_CTce_ThAb.nii.gz"
                vol_dir=os.path.join(fullpath,vol_filename)
                seg_dir=os.path.join(segpath,directory)
                #load images with nibabels
                mask=nib.load(seg_dir)
                full=nib.load(vol_dir)
                #merge, englob, and append to list
                mer=merge(mask,full)
                mer=bbox2_3D(mer)
                affine_sum=affine_sum+mask.affine
                sumofall=add_two(sumofall,mer)
                sumofall=bbox2_3D(sumofall)
        sumofall=np.dot(1/nb,sumofall)
        affine_sum = np.dot(1 / nb, affine_sum)
        sumofall=gaussian_filter(sumofall, sigma=5)
        sumofall=bbox2_3D(sumofall)
        path_to_save=os.path.join('/media/tom/TOSHIBA EXT/PIR',excluded_patient,classe+'.nii.gz')
        img = nib.Nifti1Image(sumofall, affine_sum)
        img.to_filename(path_to_save)
        #nib.save(sumofall,path_to_save )
        print("done with ",classe)

def get_classes():
    return classes


def exact_specific():
    full=nib.load("/media/tom/TOSHIBA EXT/visceral/volumes/CTce_ThAb/10000100_1_CTce_ThAb.nii.gz")
    mask = nib.load("/media/tom/TOSHIBA EXT/visceral/segmentations/CTce_ThAb/10000100_1_CTce_ThAb_58_6.nii.gz")
    mer = merge(mask, full)
    mer = bbox2_3D(mer)
    path_to_save = '/media/tom/TOSHIBA EXT/PIR/100/exact_58.nii.gz'
    img = nib.Nifti1Image(mer, full.get_affine())
    img.to_filename(path_to_save)

def as_segmentation(excluded_patient):
    try:
        os.mkdir(os.path.join('/media/tom/TOSHIBA EXT/PIR',excluded_patient))
    except Exception as e: print(e)

    valid_dirs=[]
    for directory in segfiles:
        if get_patient(directory)!=excluded_patient and get_patient(directory) in patients:
            valid_dirs.append(directory)
    for classe in classes:
        print("starting class",classe)
        sumofall=np.zeros((512,512,413))
        affine_sum=np.zeros((4,4))
        nb=0
        for directory in valid_dirs:
            if(get_class(directory)==classe):
                nb+=1
                print("\t patient"+get_patient(directory))
                vol_filename="10000"+get_patient(directory)+"_1_CTce_ThAb.nii.gz"
                vol_dir=os.path.join(fullpath,vol_filename)
                seg_dir=os.path.join(segpath,directory)
                #load images with nibabels
                mask=nib.load(seg_dir)
                full=nib.load(vol_dir)
                #merge, englob, and append to list
                mer=merge(mask,full)
                mer=bbox2_3D(mer)
                affine_sum=affine_sum+mask.affine
                sumofall=add_two(sumofall,mer)
                #sumofall=bbox2_3D(sumofall)
        sumofall=np.dot(1/nb,sumofall)
        affine_sum = np.dot(1 / nb, affine_sum)
        sumofall=gaussian_filter(sumofall, sigma=5)
        path_to_save=os.path.join('/media/tom/TOSHIBA EXT/PIR',str(excluded_patient),str(classe)+'as_seg.nii.gz')
        img = nib.Nifti1Image(sumofall, affine_sum)
        img.to_filename(path_to_save)
        #nib.save(sumofall,path_to_save )
        print("done with ",classe)
if __name__=="__main__":
    for patient in patients_to_redo:
        make_templates(patient)
        #exact_specific()
    #as_segmentation(100)
    #make_templates(100)
