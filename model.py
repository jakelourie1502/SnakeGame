####Create model to predict Q(S,A)
import torch
import numpy as np
import random
from random import randint

'''
In: Input, 

+Image
-----> 3 x conv layer
-----> Dense Layer

Out: 3 x Outputs, one for each option
'''

class SnakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #general use
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        ######Convolutional Layers
        self.conv1 = torch.nn.Conv2d(2,16, kernel_size = 2, stride = 1,padding=(0,0)); torch.nn.init.xavier_uniform_(self.conv1.weight) #19
        self.conv2 = torch.nn.Conv2d(16,32, kernel_size = 3, stride = 2,padding=(1,1)); torch.nn.init.xavier_uniform_(self.conv2.weight) #10
        self.conv3 = torch.nn.Conv2d(32,64, kernel_size = 3, stride = 2,padding=(1,1)); torch.nn.init.xavier_uniform_(self.conv3.weight) #5
        self.flatten = torch.nn.Flatten()
        
        ######Dense Layers
        self.DenseLayer1 = torch.nn.Linear(1600,3); torch.nn.init.xavier_uniform_(self.DenseLayer1.weight)

        
    def forward(self,image): #direction removed
        #conv layer
        x1 = self.conv1(image); x1=self.relu(x1)
        x2 = self.conv2(x1); x2=self.relu(x2)
        x3 = self.conv3(x2); x3=self.relu(x3)
        x4 = self.flatten(x3)

        #dense
        output = self.DenseLayer1(x4); output = self.tanh(output)
        return output
    
    def pick_move(self, image, mode='tiny_epsilon',epsilon=0.2):
        '''
        In: 1x2x20x20
        Out: 0,1,2
        '''
        if mode=='tiny_epsilon':
          random_or_not = np.random.uniform(0,1)
          if random_or_not < epsilon:
              return randint(0,2)
          image = image.reshape(1,2,20,20)
          Qvalues = self.forward(image)
          return torch.argmax(Qvalues,dim=1).numpy()[0]

