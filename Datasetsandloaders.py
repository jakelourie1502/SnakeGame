import torch
import numpy as np
'''Here we create a dataset object that allows to access the different elements of our data
In: Arrays: image, DirectionPointing, compassMoves,FirstPersonMoves, Reward
Out: [All inputs in a Class that is callable]
'''

class sample_dataset(torch.utils.data.Dataset):
    def __init__(self, Image, NextImage, Moves, Reward, Done): 
        self.Image = Image.astype('float')
        self.NextImage = NextImage.astype('float')
        self.Moves = Moves.astype('float')
        self.Reward = Reward.astype('float')
        self.Done = Done.astype('float')
        # Not dependent on index+
    def __getitem__(self, index):
        return self.Image[index],self.NextImage[index],self.Moves[index], self.Reward[index], self.Done[index]

    def __len__(self):
        return len(self.Reward)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dm_max):
        self.Image = np.zeros((dm_max,2,20,20))
        self.NextImage = np.zeros((dm_max,2,20,20))
        self.Moves = np.zeros((dm_max,))
        self.Reward = np.zeros((dm_max,))
        self.Done = np.zeros((dm_max,))
        self.max_frame = 0

    # Not dependent on index+
    def __getitem__(self, index):
        return self.Image[index],self.NextImage[index],self.Moves[index], self.Reward[index], self.Done[index]

    def __len__(self):
        return len(self.Reward)
    
    def get_sample(self, sample_size):
        sample_indices = np.random.uniform(0,self.max_frame,size=int(sample_size)).astype('int')
        return self.Image[sample_indices], self.NextImage[sample_indices], self.Moves[sample_indices], self.Reward[sample_indices], self.Done[sample_indices]

    def append(self, Image, NextImage, Moves, Reward, Done, frame_start, frame_end):
        self.Image[frame_start:frame_end] = Image
        self.NextImage[frame_start:frame_end] = NextImage
        self.Moves[frame_start:frame_end] = Moves
        self.Reward[frame_start:frame_end] = Reward
        self.Done[frame_start:frame_end] = Done        
        if frame_end > self.max_frame:
            self.max_frame = frame_end

def Dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader
    # def create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size):
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, )   ### dataloader batches the data
    #     return train_dataloader
    
    # datasets, dataloaders = {}, {}
    # train_dataset, validation_dataset, test_dataset = create_TTV_splits(train_perc, val_perc, dataset)
    # train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size)
    
    # datasets['train'], datasets['val'], datasets['test'] = train_dataset, validation_dataset, test_dataset
    # dataloaders['train'], dataloaders['val'], dataloaders['test'] = train_dataloader, val_dataloader, test_dataloader
    # return datasets, dataloaders