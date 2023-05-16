import os
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as CV

from PIL import Image
import numpy as np

class CamVidDataset():
    
    def __init__(self, images_dir, masks_dir):
        self.image_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

    def __len__(self):
        return len(self.image_ids)
 
    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array( Image.open(self.masks_fps[i]).convert('RGB'))
        
        return image, mask[:,:,0]
        

# 设置数据集路径
DATA_DIR = r'/home/data/yangrq22/data/CamVid/' # 数据集根路径
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')
    
train_dataset = CamVidDataset(
    x_train_dir, 
    y_train_dir, 
)
val_dataset = CamVidDataset(
    x_valid_dir, 
    y_valid_dir, 
)


# dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"], shuffle=True)



def create_Dataset(data_path, batch_size, device_num, shuffle):
    x_train_dir = os.path.join(data_path, 'train')
    y_train_dir = os.path.join(data_path, 'train_labels')
    x_valid_dir = os.path.join(data_path, 'val')
    y_valid_dir = os.path.join(data_path, 'val_labels')
    train_dataset = CamVidDataset(x_train_dir, y_train_dir)
    val_dataset = CamVidDataset(x_valid_dir, y_valid_dir)
    operations = [CV.Resize((224, 224)), CV.HorizontalFlip(), CV.VerticalFlip(), CV.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]), CV.HWC2CHW()]
    data_set = ds.GeneratorDataset(train_dataset, column_names=["image", "mask"], shuffle=shuffle, num_shards=device_num, shard_id=0)
    data_set = data_set.map(input_columns=["image"], operations=operations)
    data_set = data_set.map(input_columns=["mask"], operations=[CV.HWC2CHW()])
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, data_set.get_dataset_size()


# print(dataset.get_dataset_size())

dataset, dataset_size = create_Dataset(DATA_DIR, 1, 1, True)
iterator = dataset.create_dict_iterator()
for data_dict in iterator:
    for name in data_dict.keys():
        print(name, data_dict[name].shape)
    print("="*20)