import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import torch
import cv2
"""

covid ct image the maximum number of cr assumed 70

Parameters:

-t (--text): show the text interface
-h (--help): display this help
"""
csv_file = '/home/tookai-/covid_safavi/Multimodal-Prognosis-COVID-19-/dataset/label.csv'
a= pd.read_csv(csv_file).iloc[:,:]
class covid_ct(Dataset):

    def __init__(self, root, image_dir, csv_file, transform=None):
        self.root = root
        self.image_dir = image_dir
        self.image_folders = glob.glob('/home/tookai-/covid_safavi/Multimodal-Prognosis-COVID-19-/dataset/ct/*')
        self.transform = transform
        self.data = pd.read_csv(csv_file)
       



    def __len__(self):
        print("len",len(self.data))
        return len(self.data)-1

    def __getitem__(self, index):
        print("index",index,len(self.image_folders))
        image_name = self.image_folders[index]
        image_list = glob.glob(image_name+'/lung_white/*.jpg')
        covid_images = []
        print(len(image_list))
        v = torch.ones([80,512,512,3]) 
        i = 0
        
        patiant_id = image_name.split('/')[-1]
        for x in image_list:
            image1 = cv2.imread(x)
            img = torch.from_numpy(image1)

            v[i,:,:,:] =img
            i = i+1
        #get label
        print("--------",patiant_id)
        print(self.data.query('id==@patiant_id')['y'],"************************************")
        label = self.data.query('id==@patiant_id')['y'].iloc[0]

        return (v,label)