
import skimage.io as io
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from multiprocessing import Process, Array,Pool
import nibabel as nib
import random
import matplotlib.patches as patches
from scipy.signal import correlate as correlate
from scipy.signal import fftconvolve as fftconvolve
import time
from skimage.transform import pyramid_gaussian,downscale_local_mean
import gc
from templateMaker import bbox2_3D,broadcast_2d,broadcast_3d,get_patient


def display_square(image,tshape, maxind,solution=None):
    x_t,y_t,z_t=tshape
    fig, [ax1 , ax2 , ax3] = plt.subplots(1, 3, num='Result of Template Search')
    rectxy = patches.Rectangle((maxind[1] - int(y_t / 2), maxind[0] - int(x_t / 2)), y_t, x_t, linewidth=1,
                             edgecolor='r', facecolor='none')
    rectyz = patches.Rectangle((maxind[2] - int(z_t / 2), maxind[1] - int(y_t / 2)), z_t, y_t, linewidth=1,
                               edgecolor='r', facecolor='none')
    rectxz = patches.Rectangle((maxind[2] - int(z_t / 2), maxind[0] - int(x_t / 2)), z_t, x_t, linewidth=1,
                               edgecolor='r', facecolor='none')

    ax1.imshow(image[:,:,maxind[2]], interpolation='nearest')
    ax1.add_patch(rectxy)
    ax2.imshow(image[maxind[0], :,:], interpolation='nearest')
    ax2.add_patch(rectyz)
    ax3.imshow(image[:, maxind[1], :], interpolation='nearest')
    ax3.add_patch(rectxz)
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




def pyr(pyramid,template,layer,max_recursion,ind=(0.0,0.0,0.0)):
    #print("couche ",layer)
    if layer<1:
        #print("last layer")
        image=pyramid[0]
        oneachside=50
        cx,cy,cz=ind
        xinf,xsup=int(cx * image.shape[0] - oneachside),int(cx * image.shape[0] + oneachside),
        yinf,ysup=int(cy * image.shape[1] - oneachside), int(cy * image.shape[1] + oneachside),
        zinf,zsup=int(cz * image.shape[2] - oneachside), int(cz * image.shape[2] + oneachside)

        last_region=image[xinf:xsup,yinf:ysup,zinf:zsup]
        nxcorr = fft_nxcorr(f=last_region, t=template)
        corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
        realind = (xinf + corrind[0], yinf + corrind[1], zinf + corrind[2])
        print("\trealind", realind)
        #print("\tcorr index", corrind)
        #print("\tnxcorr max", np.amax(nxcorr))
        print("\tcorr max",np.amax(nxcorr))
        return realind

    couche=pyramid[layer]
    #print("\tcouche shape",couche.shape)
    #print('relative ind passed to this layer is', ind )
    max_layer=len(pyramid)-1

    #Ratio in every dimension between image and template dimensions
    #NOTE : this doesn't change during the whole procedure
    ratios=(template.shape[0]/pyramid[0].shape[0],
            template.shape[1]/pyramid[0].shape[1],
            template.shape[2]/pyramid[0].shape[2])
    #We multiply by the image of layer's dims
    resized_shape=(int(ratios[0]*couche.shape[0]),
                   int(ratios[1]*couche.shape[1]),
                   int(ratios[2]*couche.shape[2]))
    downscaling_factor=(pyramid[0].shape[0] // pyramid[layer].shape[0],
                        pyramid[0].shape[1] // pyramid[layer].shape[1],
                        pyramid[0].shape[2] // pyramid[layer].shape[2])
    #Here's where we downscale
    r_t=downscale_local_mean(template,downscaling_factor)
    #display(couche, r_t)
    #print("\tresied template shape",r_t.shape)
    #print("\texected shape is",resized_shape)
    #We now have to calculate the borders region of interest
    #It is based on previous iterations (or not if it is the first one)
    if(layer==max_recursion):
        #the layer is the maximum one we need togo through the whole image
        nxcorr = fft_nxcorr(f=couche, t=r_t)
        corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
        #print("\t corr index",corrind)
        realind = (corrind[0], corrind[1], corrind[2])
    else:
        #We have to calculate what is the absolute index in our case
        abs_ind=(int(couche.shape[0]*ind[0]),
                 int(couche.shape[1]*ind[1]),
                 int(couche.shape[2]*ind[2]))
        #print("\tabs ind",abs_ind)
        #We use a parameter sigma that will determine how big the region is
        #Let's establish that if sigma = 1 the RoI is of one template
        sigma=1.5
        xinf,xsup=abs_ind[0]-int(sigma*resized_shape[0]/2),abs_ind[0]+int(sigma*resized_shape[0]/2)
        yinf, ysup = abs_ind[1] - int(sigma * resized_shape[1] / 2), abs_ind[1] + int(sigma * resized_shape[1] / 2)
        zinf, zsup = abs_ind[2] - int(sigma * resized_shape[2] / 2), abs_ind[2] + int(sigma * resized_shape[2] / 2)
        #We readjust the border just in case
        if xinf<0:xinf=0
        if yinf<0:yinf=0
        if zinf<0:zinf=0
        if xsup > couche.shape[0]:xsup=couche.shape[0]-1
        if ysup > couche.shape[1]: xsup = couche.shape[1] - 1
        if zsup > couche.shape[2]: xsup = couche.shape[2] - 1
        #print("\t RoI shape ",couche[xinf:xsup,yinf:ysup,zinf:zsup].shape)
        #We correlate with a view of the image rather than another object
        #to have the max in the coordinates of the image
        nxcorr=fft_nxcorr(f=couche[xinf:xsup,yinf:ysup,zinf:zsup],t=r_t)
        corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
        realind = (xinf + corrind[0], yinf + corrind[1], zinf + corrind[2])
        #print("corr index",corrind)
        #print("\tnxcorr max",np.amax(nxcorr))
        #display(nxcorr,r_t)
    # Find max in corr
    corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
    #print("max in corr at",corrind)

    #print("\trealind",realind)
    relative_ind=(realind[0] / couche.shape[0], realind[1] / couche.shape[1], realind[2] / couche.shape[2])

    return pyr(pyramid, template, layer - 1,max_recursion,relative_ind)




def fft_nxcorr(f,t,array=None,i=None):
    """
        This method is heavily influenced (and in some occasions copied)  by the code of

        Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>

        that can be found here :
        https://pastebin.com/x1NJqWWm

        referencing:
            Hermosillo et al 2002: Variational Methods for Multimodal Image
            Matching, International Journal of Computer Vision 50(3),
            329-343, 2002
            <http://www.springerlink.com/content/u4007p8871w10645/>

            Lewis 1995: Fast Template Matching, Vision Interface,
            p.120-123, 1995
            <http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html>

            <http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation>

    """
    t = np.float32(t)
    f = np.float32(f)

    std_t, mean_t,tsize= np.std(t), np.mean(t),t.size
    xcorr = correlate(f, t,'same','fft')

    """
    Rather than calculate integral tables to get the local sums, we will convolve by an array of ones, that has the shape
    of the template. using fft again
    """
    convolver=np.ones_like(t)
    ls_a=fftconvolve(f,convolver,'same')
    ls2_a = fftconvolve(f ** 2, convolver, 'same')

    # local standard deviation of the input array
    ls_diff = ls2_a - (ls_a ** 2) / tsize
    ls_diff = np.where(ls_diff < 0, 0, ls_diff)
    sigma_a = np.sqrt(ls_diff)

    # standard deviation of the template
    sigma_t = np.sqrt(t.size - 1.) * std_t

    # denominator: product of standard deviations
    denom = sigma_t * sigma_a

    # numerator: local mean corrected cross-correlation
    numer = (xcorr - ls_a * mean_t)

    # sigma_t cannot be zero, so wherever the denominator is zero, this must
    # be because sigma_a is zero (and therefore the normalized cross-
    # correlation is undefined), so set nxcorr to zero in these regions
    tol = np.sqrt(np.finfo(denom.dtype).eps)
    nxcorr = np.where(denom < tol, 0, numer / denom)

    # if any of the coefficients are outside the range [-1 1], they will be
    # unstable to small variance in a or t, so set them to zero to reflect
    # the undefined 0/0 condition
    nxcorr = np.where(np.abs(nxcorr - 1.) > np.sqrt(np.finfo(nxcorr.dtype).eps), nxcorr, 0)
    if array!=None:
        array[i]=np.amax(nxcorr)
    #time.sleep(random.randint(1,3))
    return nxcorr



def load_images(fullpath='/media/tom/TOSHIBA EXT/visceral/volumes/CTce_ThAb/10000131_1_CTce_ThAb.nii.gz'
                ,templatepath='/media/tom/TOSHIBA EXT/PIR/131/1302.nii.gz'):
    print("loading image", fullpath)
    image = nib.load(fullpath)
    print("loading template", templatepath)
    template = nib.load(templatepath)

    return image,template



