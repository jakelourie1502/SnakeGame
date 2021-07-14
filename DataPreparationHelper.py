'''Set of functions that convert the snake data into usable arrays

In: Dictionary -----> data, ArrayMax(estimate_of_size), Reward parameters. 
Out: Image of 2 frames, New state, CompassMoves, FirstPersonMoves, reward, done ---> Arrays
'''

######  Master Function that calls all functions below
import numpy as np
import torch

def CreateAllXYInputs(data, ArrayMax):

    ###Helper: The Dictionary keys that list the moves made are strings. To look at them sequentially we need to them to be integers
    def TurnDictKeysToInt(data,instance,dic):
        return {int(k):v for k,v in data[instance][dic].items()}

    def CreateImagesAllGames(ArrayMax, Data):
        ###Helper function that convers the body and food values for ONE game.
        def CreateOneGameOneObj(bodyimage, foodimage, image, index,next=False):
            for key, body in bodyimage.items():
                for idx, point in enumerate(body):
                    y = point[0]-1 ### when we 'view' our game like the snake.py, y is at the top
                    x = point[1]-1 ### In our snake game, the arrays start at 1, and go to 20. We want 0->19
                    if next==True and (y < 0 or y>=20 or x<0 or x>=20):
                        pass
                    else: 
                        image[int(key),index,y,x] = 0.5
            
            for key, value in foodimage.items():    #food
                y = value[0]-1 #-1 because the stored array starts at 1, instead of 0
                x = value[1]-1
                image[int(key),index,y,x] =1
            return image

        def CreateImages(instance,data): ###Array Max is how big we set the initial array of all zeros, that gets populated.
            IntegerKeysDictNowBody = TurnDictKeysToInt(data,instance,'body')
            IntegerKeysDictNowFood = TurnDictKeysToInt(data,instance,'FoodPos')
            IntegerKeysDictPreviousBody = TurnDictKeysToInt(data,instance,'previous_body')        
            IntegerKeysDictPreviousFood = TurnDictKeysToInt(data,instance,'FoodPosPrev')
            IntegerKeysDictNextBody = TurnDictKeysToInt(data,instance,'new_body')
            IntegerKeysDictNextFood = TurnDictKeysToInt(data,instance,'FoodPosNext')
            plays = len(IntegerKeysDictNowBody)
            image = np.zeros((plays,2,20,20)).astype('float')
            NextImage = np.zeros((plays,2,20,20)).astype('float')
            #Channel 1: now, channel 2: previous
            image = CreateOneGameOneObj(IntegerKeysDictNowBody, IntegerKeysDictNowFood, image, 1)
            image = CreateOneGameOneObj(IntegerKeysDictPreviousBody, IntegerKeysDictPreviousFood, image, 0)
            NextImage = CreateOneGameOneObj(IntegerKeysDictNowBody, IntegerKeysDictNowFood, NextImage, 0, next=False)
            NextImage = CreateOneGameOneObj(IntegerKeysDictNextBody, IntegerKeysDictNextFood, NextImage, 1, next=True)

            return image, NextImage
        
        AllGamesImage = np.zeros((ArrayMax,2,20,20))
        AllGamesNextImage = np.zeros((ArrayMax,2,20,20))
        frames_start=0 ###We need to keep a track of how many frames we have in case we use more than the size of the array we specified in array max.
        for game in data.keys():
            ImagetempArray, NextImagetempArray = CreateImages(game,data)
            frames_end = frames_start+ (len(ImagetempArray)) #to create start and end for storing the x values
            #if there are two many values compared to max array, we end the function there.
            if frames_end > ArrayMax:
                return AllGamesImage, AllGamesNextImage, frames_start
            AllGamesImage[frames_start:frames_end] = ImagetempArray
            AllGamesNextImage[frames_start:frames_end] = NextImagetempArray
            frames_start = frames_end #set frames start to frames end for next iteration
        return AllGamesImage, AllGamesNextImage, frames_start

    def RewardDoneFPChoiceJoinAllGames(ArrayMax, data, metric):
        
        AllGamesJoined = np.zeros((ArrayMax))
        frames_start = 0
        for game in data.keys():
            IntegerKeysDictOneGame = TurnDictKeysToInt(data,game,metric)
            ArrayOneGame = np.array(list(IntegerKeysDictOneGame.values())).astype('float')
            frames_end = frames_start + len(ArrayOneGame)
            if frames_end > ArrayMax:
                return AllGamesJoined, frames_start
            AllGamesJoined[frames_start:frames_end] = ArrayOneGame
            frames_start = frames_end
        return AllGamesJoined, frames_start

    AllGamesImage, AllGamesNextImage, frames1 = CreateImagesAllGames(ArrayMax, data)
    FirstPersonMoves, frames2 = RewardDoneFPChoiceJoinAllGames(ArrayMax,data,'FirstPersonChoice')
    Reward, frames3 = RewardDoneFPChoiceJoinAllGames(ArrayMax,data,'Score')
    Done, frames4 = RewardDoneFPChoiceJoinAllGames(ArrayMax,data,'Dead')
    AllGamesImage, AllGamesNextImage, FirstPersonMoves,Reward,Done = AllGamesImage[:frames1], AllGamesNextImage[:frames1], FirstPersonMoves[:frames1],Reward[:frames1],Done[:frames1]
    return AllGamesImage, AllGamesNextImage, FirstPersonMoves, Reward, Done, frames1, frames2, frames3, frames4 

'''This is for converting just one set of body and previous body (& food) to an image that can be used to make a move choice'''
def CreateOneFrameForChoosingMove(body, prev_body, food, prev_food):
      image = np.zeros((1,2,20,20))
      for idx, point in enumerate(body):
          y = point[0]-1 ### when we 'view' our game like the snake.py, y is at the top
          x = point[1]-1 ### In our snake game, the arrays start at 1, and go to 20. We want 0->19
          image[0,1,y,x] = 0.5
      
      for idx, point in enumerate(prev_body):
        y = point[0]-1 ### when we 'view' our game like the snake.py, y is at the top
        x = point[1]-1 ### In our snake game, the arrays start at 1, and go to 20. We want 0->19
        image[0,0,y,x] = 0.5
      
      y = food[0]-1 #-1 because the stored array starts at 1, instead of 0
      x = food[1]-1
      image[0,1,y,x] =1

      y = prev_food[0]-1 #-1 because the stored array starts at 1, instead of 0
      x = prev_food[1]-1
      image[0,0,y,x] =1
      return torch.Tensor(image)

def get_samples(sample_percentage, AllGamesImage, AllGamesNextImage, FirstPersonMoves, Reward, Done):
    sample_indices = (np.random.uniform(0,len(AllGamesImage),size=int(sample_percentage*len(AllGamesImage)))).astype('int')
    def get_sample_1_metric(metric, sample_indices):
        return metric[sample_indices]
    sample_Image = get_sample_1_metric(AllGamesImage, sample_indices)
    sample_NextImage = get_sample_1_metric(AllGamesNextImage, sample_indices)
    sample_Moves = get_sample_1_metric(FirstPersonMoves, sample_indices)
    sample_Reward = get_sample_1_metric(Reward, sample_indices)
    sample_Done = get_sample_1_metric(Done, sample_indices)
    return sample_Image, sample_NextImage, sample_Moves, sample_Reward, sample_Done

