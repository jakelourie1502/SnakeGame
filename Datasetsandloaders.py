import torch
'''Here we create a dataset object that allows to access the different elements of our data
In: Arrays: image, DirectionPointing, compassMoves,FirstPersonMoves, Reward
Out: [All inputs in a Class that is callable]
'''

class Dataset(torch.utils.data.Dataset):
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

'''
Take our Dataset, split the data, and put them into dataloaders
In: Dataset class, Train split, val split, batchsize
Out: [Dictionary 1: train dataset, val dataset, test dataset
      Dictionary 2: train dataloader, val dataloader, test dataloader 
    ]
'''

def DatasetsAndDataloaders(dataset, train_perc, val_perc, batch_size):
    
    def create_TTV_splits(train_perc, val_perc,dataset):
        assert train_perc + val_perc < 1, 'val and train percent should add up to <1'
        length = len(dataset)
        trainSize = int(train_perc * length)
        TestValSize = int(length - trainSize)
        valSize = int(val_perc * length)
        TestSize = int(TestValSize - valSize)

        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [trainSize, TestValSize])  ## split into 1000 training & 797 validation
        validation_dataset, test_dataset  = torch.utils.data.random_split(
            validation_dataset, [valSize, TestSize])  ## get test set from validation set

        return train_dataset, validation_dataset, test_dataset

    def create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, )   ### dataloader batches the data
        val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader
    
    datasets, dataloaders = {}, {}
    train_dataset, validation_dataset, test_dataset = create_TTV_splits(train_perc, val_perc, dataset)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size)
    
    datasets['train'], datasets['val'], datasets['test'] = train_dataset, validation_dataset, test_dataset
    dataloaders['train'], dataloaders['val'], dataloaders['test'] = train_dataloader, val_dataloader, test_dataloader
    return datasets, dataloaders