import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
"""

covid ct image the maximum number of cr assumed 70

Parameters:

-t (--text): show the text interface
-h (--help): display this help
"""
csv_file = '/home/tookai-/covid_safavi/sample_csv.csv'
a= pd.read_csv(csv_file).iloc[:,:]
class covid_ct(Dataset):

    def __init__(self, root, image_dir, csv_file, transform=None):
        self.root = root
        self.image_dir = image_dir
        self.image_folders = glob.glob('/home/tookai-/dataset/sample/*')
        self.transform = transform
        self.data = pd.read_csv(csv_file).iloc[:, :]
       



    def __len__(self):
        return 8

    def __getitem__(self, index):
        image_name = self.image_folders[index]
        image_list = glob.glob(image_name+'/lung_white/*.jpg')
        covid_images = []
        v = torch.ones([70,512,512,3]) 
        i = 0
        
        patiant_id = image_name.split('/')[-1]
        for x in image_list:
            image1 = cv2.imread(x)
            img = torch.from_numpy(image1)

            v[i,:,:,:] =img
            i = i+1
        #get label
        
        label = self.data.query('id==@patiant_id')['y'].iloc[0]

        return (v,label)