import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import scipy
from util import load_images,pyr,fft_nxcorr
from skimage.transform import pyramid_gaussian,downscale_local_mean
import skimage.io as io
import csv
import numpy as np
import math
import statistics
import time
from scipy.signal import correlate as correlate
from templateMaker import get_patient

right_center_1302_100=(354,256,326)
right_center_58_100=(313,256,227)
def display(image,template):
    fig, [[ax1, ax2, ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, num='Result of Template Search')
    ax1.imshow(image[:, :, int(image.shape[2]/2)], interpolation='nearest')
    ax1.set_title('image XY')
    ax2.imshow(image[int(image.shape[0]/2), :, :], interpolation='nearest')
    ax2.set_title('image YZ')
    ax3.imshow(image[:,int(image.shape[1]/2), :], interpolation='nearest')
    ax3.set_title('image XZ')
    ax4.imshow(template[:, :, int(template.shape[2] / 2)], interpolation='nearest')
    ax4.set_title('template XY')
    ax5.imshow(template[int(template.shape[0] / 2), :, :], interpolation='nearest')
    ax5.set_title('template YZ')
    ax6.imshow(template[:, int(template.shape[1] / 2), :], interpolation='nearest')
    ax6.set_title('template XZ')
    plt.show()

def display_square(image,tshape, maxind,solution=None):
    x_t,y_t,z_t=tshape
    fig, [ax1 , ax2 , ax3] = plt.subplots(1, 3, num='Result of Template Search')
    """
    rectxy = patches.Rectangle((maxind[1] - int(y_t / 2), maxind[0] - int(x_t / 2)), y_t, x_t, linewidth=1,
                             edgecolor='r', facecolor='none')
    rectyz = patches.Rectangle((maxind[2] - int(z_t / 2), maxind[1] - int(y_t / 2)), z_t, y_t, linewidth=1,
                               edgecolor='r', facecolor='none')
    rectxz = patches.Rectangle((maxind[2] - int(z_t / 2), maxind[0] - int(x_t / 2)), z_t, x_t, linewidth=1,
                               edgecolor='r', facecolor='none')
    ax1.add_patch(rectxy)
    ax2.add_patch(rectyz)
    ax3.add_patch(rectxz)
    """
    ax1.imshow(image[:,:,maxind[2]], interpolation='nearest',cmap='Greys_r')
    ax2.plot(maxind[0], maxind[1], 'r+')

    ax2.imshow(image[maxind[0], :,:], interpolation='nearest',cmap='Greys_r')
    ax1.plot(maxind[1], maxind[2], 'r+')

    ax3.imshow(image[:, maxind[1], :], interpolation='nearest',cmap='Greys_r')
    ax3.plot(maxind[0], maxind[2], 'r+')

    if solution!=None:
        x_s,y_s,z_s=solution
        solxy = patches.Rectangle((maxind[1] - int(y_t / 2), x_s - int(x_t / 2)), y_t, x_t, linewidth=1,
                                   edgecolor='g', facecolor='none')
        solyz = patches.Rectangle((maxind[2] - int(z_t / 2), y_s - int(y_t / 2)), z_t, y_t, linewidth=1,
                                   edgecolor='g', facecolor='none')
        solxz = patches.Rectangle((maxind[2] - int(z_t / 2), z_s - int(x_t / 2)), z_t, x_t, linewidth=1,
                                   edgecolor='g', facecolor='none')
        ax1.add_patch(solxy)
        ax2.add_patch(solyz)
        ax3.add_patch(solxz)
    plt.show()

def routine_3d():
    print("3d routine starting")
    image,template=load_images()
    #display(image.get_fdata(),template.get_fdata())

    image_shape=image.get_fdata().shape
    pyramid = tuple(pyramid_gaussian(image.get_fdata(), downscale=2, multichannel=False))
    max_layer = len(pyramid) - 1
    #template_rotate= np.rot90(template.get_fdata(), k=1)
    #ind=shotgun(image.get_fdata(),template.get_fdata(),3)
    t_dep=time.clock()
    ind = pyr(pyramid=pyramid, template=template.get_fdata(), layer=max_layer -5,max_recursion=max_layer-5)
    print("operation performed",time.clock()-t_dep)
    #ind=shotgun(image.get_fdata(),template.get_fdata(),3)
    print("index is",ind)
    print("or in world coordinates : ",world_coord(image,ind[0],ind[1],ind[2]))
    display_square(image.get_fdata(),template.shape,ind)

def world_coord(img,i, j, k):
    M = img.affine[:3, :3]
    abc = img.affine[:3, 3]
    return M.dot([i, j, k]) + abc

def distance(prediction,factual):
    if len(prediction)!=len(factual):
        print("prediction and factual should have same length")
        return -1
    d=0
    for i in range(len(prediction)):
        d += (float(prediction[i])-float(factual[i]))**2
    return math.sqrt(d)

def test_classe(classe,templates_path='/media/tom/TOSHIBA EXT/PIR',fullpath='/media/tom/TOSHIBA EXT/visceral/volumes/CTce_ThAb'):
    good=0
    fair=0
    bad=0
    results=[]
    print("testing classe",classe)
    verif_path="/media/tom/TOSHIBA EXT/centers"
    classe_template_name=str(classe)+".nii.gz"
    all_patients = os.listdir(templates_path)
    with open(str(classe)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["patient", "i", "j","k","X","Y","Z"])
        for imagename in os.listdir(fullpath):
            try:
                image,template=load_images(fullpath=os.path.join(fullpath,imagename),templatepath=os.path.join(templates_path,get_patient(imagename),classe_template_name))
                pyramid = tuple(pyramid_gaussian(image.get_fdata(), downscale=2, multichannel=False))
                max_layer = len(pyramid) - 1
                ind = pyr(pyramid=pyramid, template=template.get_fdata(), layer=max_layer-5,max_recursion=max_layer-5)
                w=world_coord(image,ind[0],ind[1],ind[2])
                print("\tor in world coordinates : ", w)
                #inflated_index = (int(image.shape[0] * ind[0]), int(image.shape[1] * ind[1]), int(image.shape[2] * ind[2]))
                writer.writerow([get_patient(imagename),ind[0],ind[1],ind[2],w[0],w[1],w[2]])
                filename='10000'+get_patient(imagename)+'_1_CTce_ThAb_'+str(classe)+'_center.csv'
                with open(os.path.join(verif_path,filename), 'r') as verif_reader:
                    reader = csv.reader(verif_reader)
                    for row in reader:
                        verif_voxel=(row[0],row[1],row[2])
                        verif_world=(row[3],row[4],row[5])
                        distance_v=distance(ind, verif_voxel)
                        distance_w=distance(w, verif_world)
                        results.append(distance_w)
                        print("\tvoxelic distance to verification ",distance_v)
                        print("\tworld distance to verification ", distance_w)
                        if distance_w<=30:
                            good+=1
                        elif distance_w<=80:
                            fair+=1
                        else:
                            bad+=1
            except:
                print("fucked up here")
    with open("resultats.csv","w") as csvfile:
        writer=csv.writer(csvfile, delimiter=' ',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['classe', 'good', 'fair', 'bad'])
        writer.writerow([str(classe),str(good),str(fair),str(bad)])
        writer.writerow(['mean','median','std'])
        writer.writerow([statistics.mean(results), statistics.median(results),statistics.stdev(results)])

def display_lung_test():
    image,template=load_images()
    image=image.get_fdata()
    template=template.get_fdata()
    display_square(image,template.shape,right_center_58_100)


def routine_2d():
    image = io.imread('/home/tom/Images/autres/berners.jpg', as_gray=True)
    template=image[270:300,850:900]
    i_row, i_col = image.shape
    print(image.shape)
    t_row, t_col = template.shape
    t_dep = time.clock()
    nxcorr=fft_nxcorr(image,template)
    correlation=correlate(image,template,'same','fft')
    maxind_nxcorr = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
    maxind_corr = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)
    print("correlation construite en ", time.clock() - t_dep)
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, num='ND Template Search')
    ax1.imshow(image, interpolation='nearest',cmap='Greys_r')
    ax1.set_title('Search image')
    ax2.imshow(template, interpolation='nearest',cmap='Greys_r')
    ax2.set_title('Template')
    print(maxind_nxcorr)
    ax3.imshow(correlation,interpolation='nearest',cmap='Greys_r')
    ax3.set_title('Regular cross-correlation')
    ax4.imshow(nxcorr, interpolation='nearest',cmap='Greys_r')
    ax4.set_title('Normalized cross-correlation')
    rect_corr = patches.Rectangle((maxind_corr[1] - int(t_col / 2), maxind_corr[0] - int(t_row / 2)), t_col, t_row,
                             linewidth=1,
                             edgecolor='r', facecolor='none')
    rect = patches.Rectangle((maxind_nxcorr[1] - int(t_col / 2), maxind_nxcorr[0] - int(t_row / 2)), t_col, t_row, linewidth=1,
                             edgecolor='r', facecolor='none')
    ax4.add_patch(rect)
    ax3.add_patch(rect_corr)
    plt.show()


if __name__=='__main__':
    #multi()
    #performance()
    #routine_3d()
    #routine_2d()
    #perfo_test()
    #display_lung_test()
    test_classe(1302)
    #test_classe(1326)
    #test_classe(30324)


