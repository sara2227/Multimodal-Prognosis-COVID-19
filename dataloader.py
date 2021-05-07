import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import torch
import cv2
import numpy as np
from scipy.ndimage import zoom
"""

covid ct image the maximum number of cr assumed 70

Parameters:

-t (--text): show the text interface
-h (--help): display this help
"""


''' fix ct number'''
# Resize 2D slices
w, h = 512, 512
def rs_img(img):
    '''W and H is 128 now
    '''
#     print("type",type(img))
    img = np.transpose(img)
#     print(img.shape,type(img))
    flatten = [cv2.resize(img[:,:,i], (w, h), interpolation=cv2.INTER_CUBIC) for i in range(img.shape[-1])]
    img = np.array(np.dstack(flatten)) 
    return img

# Spline interpolated zoom (SIZ)
def change_depth_siz(img):
    desired_depth = 64
    current_depth = img.shape[-1]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (1, 1, depth_factor), mode='nearest')
    return img_new

def normalize(image):
    global MIN_BOUND
    global MAX_BOUND
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def plot_seq(data, name):
    
    a, b = 3, 20
    data = np.reshape(data, (a, b, 512, 512))
    test_data = data
    r, c = test_data.shape[0], test_data.shape[1]

    cmaps = [['viridis', 'binary'], ['plasma', 'coolwarm'], ['Greens', 'copper']]

    heights = [a[0].shape[0] for a in test_data]
    widths = [a.shape[1] for a in test_data[0]]

    fig_width = 10.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)

    f, axarr = plt.subplots(r,c, figsize=(fig_width, fig_height),
          gridspec_kw={'height_ratios':heights})

    for i in range(r):
        for j in range(c):
            axarr[i, j].imshow(test_data[i][j], cmap='gray')
            axarr[i, j].axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig('{}/{}.png'.format('.', name), dpi=1000)
    plt.show()


# NOT USING THIS NOW
def resize_depth_wise(img3d):
    '''
      Inputs a 3d tensor with uneven depth
      Outputs a 3d tensor with even depth, in this case depth=64
    '''
# patient image 3D
    p = img3d
# list of 3D slices of p
    p_2d = []

    depth = 2

    n = 0
    c = 0

    for c in range(70):
        img = img3d[:,:,n+depth]
        p_2d.append(img)
        n = n+depth
        c = c+1
        
    p_3d_d64 = np.array(np.dstack(p_2d))
    return p_3d_d64
''' covid class data loader
	input: ad
'''
class covid_ct(Dataset,):

    def __init__(self, root, csv_file):
        self.root = root
        self.image_folders = glob.glob('/home/tookai-1/Desktop/sara/covid_safavi/code01/dataset/edited/*')
        # print(csv_file)
        self.data = pd.read_csv(csv_file).iloc[:, :]

    def __len__(self):

        return len(self.data )
    

    def __getitem__(self, index):
        # print(self.image_folders)
        image_name = self.image_folders[index]
        image_list = glob.glob(image_name+'/lung_white/*.jpg')
        covid_images = []
        len_imgs = len(image_list)
        v = np.ones([len_imgs,512,512]) 
        i = 0
        
        patiant_id = image_name.split('/')[-1]
        for x in image_list:
            image1 = cv2.imread(x,0)
            img = torch.from_numpy(image1)
            
            v[i,:,:] =img
            i = i+1
  
        print(patiant_id)
        label = self.data.query('id==@patiant_id')['label'].iloc[0]
        features = self.data.query('id==@patiant_id').loc[:,'men':'CRP']

        img = rs_img(v)
        img_siz = change_depth_siz(img)
        img_siz = np.transpose(img_siz)
        inputs = np.reshape(img_siz, (64,512,512,1))

        return(inputs,label,features.to_numpy())







