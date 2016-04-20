
import numpy as np
from os import listdir
from os.path import isfile, join

from scipy import misc
import matplotlib.pyplot as plt

img_mean = np.load('pretrained_weights/img_mean.npy')
print 'img_mean shape : ',img_mean.shape
print np.mean(np.mean(img_mean,axis=1),axis=1)

file_list = ['van_gogh_starry_night.jpg', 'kaist_n1.jpg']

for i in file_list: 

    f = misc.imread(i)
    min_dim, max_dim = np.argmin(f.shape[:2]), np.argmax(f.shape[:2])

    resize_scale = 227.0/f.shape[min_dim]

    f = misc.imresize(f,[int(f.shape[0]*resize_scale),int(f.shape[1]*resize_scale)])
    f = f[int((f.shape[0]-227.0)/2):int((f.shape[0]-227.0)/2)+227, int((f.shape[1]-227.0)/2):int((f.shape[1]-227.0)/2)+227, :]
    print 'image shape(before) : ',f.shape

    plt.imshow(f)
    plt.show()

    f = np.transpose(f,(2,0,1))
    print 'image shape(after) : ',f.shape

    preprocessed_img = np.asarray(f,dtype=np.float32)-img_mean[:,16:16+227,16:16+227]

    np.save(i[:len(i)-4]+'.npy',preprocessed_img)

