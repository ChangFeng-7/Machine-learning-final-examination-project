from torch.utils.data import  DataLoader,Dataset
import os
from PIL import Image

class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,transforms ,datapath = None):


        self.transforms = transforms
        self.datapath = datapath
        self.img_path = os.listdir(self.datapath)
        self.path_all = []
        for i_path in self.img_path:
            name_temp = os.listdir(self.datapath + '//'  + i_path)
            for j_temp in name_temp:
                path_temp = self.datapath + '/' + i_path + '/' + j_temp
                self.path_all.append(path_temp)

    def __getitem__(self, index):

        # path_temp = self.datapath + '\\'  + self.path_all[index]
        path_temp = self.path_all[index]
        data = Image.open(path_temp).convert('RGB')

        data = self.transforms(data)

        if path_temp.split('/')[-1][0] =='1':
            label = 1
        else:
            label = 0
        # label = int(self.data_labels.index(path_temp.split('/')[-2]))
        return data, label

    def __len__(self):
        return len(self.path_all)

    def get_name(self,index):
        return self.img_path[index]

    def get_img_path(self,index):
        return self.datapath + '\\'  + self.img_path[index]