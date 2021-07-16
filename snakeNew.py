import pickle
import json
import torch
import torchmetrics
import curses
from random import randint
import time
import numpy as np
import random
import pprint
import pickle
from model import SnakeModel
from Datasetsandloaders import Dataset, DatasetsAndDataloaders
from DataPreparationHelper import CreateAllXYInputs, CreateOneFrameForChoosingMove, get_samples
from trainsystem import fit

##############################################################################################################
##
##
##  We need a 'Snake' object which the game will use.


class Snake():
    def __init__(self, vert_bounds, hor_bounds, length=3):
        self.direction = randint(0,3) #up, right, down, left
        y = randint(vert_bounds[0],vert_bounds[1])
        x = randint(hor_bounds[0],hor_bounds[1])
        if self.direction in [0,1]:        
            self.body = [[y-i,x] if self.direction==0 else [y,x-i] for i in range(length) ]
        if self.direction in [2,3]:        
            self.body = [[y+i,x] if self.direction==2 else [y,x+i] for i in range(length) ]
        self.previous_body = self.body.copy()

    @property
    def head(self):
        return self.body[0]

    @property
    def tail(self):
        return self.body[1:]

    @property
    def length(self):
        return len(self.body)

    def HasEatenItself(self):
        if self.head in self.tail:
            return True
    
    def CheckContigous(self):
        for idx, point in enumerate(self.body[:self.length-2]):
            nextPoint = self.body[idx+1]
            assert abs((point[0]-nextPoint[0]) + (point[1] - nextPoint[1])) == 1, (self.body,self.head,self.tail) 

class SnakeGame():
    def __init__(self, Qmodel, targetModel,board_width = 20, board_height = 20, gui = False,starting_length=3, epsilon=0.95, gamma = 0.95):
        ###Initialise key aspects of the game
        self.epsilon = epsilon
        self.gamma = gamma
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui  
        self.starting_length = starting_length
        self.Qmodel = Qmodel
        self.targetModel = targetModel

        ###Initialise logging
        self.moves = 0
        self.score = 0
        self.AtLeast1Score = False
        self.metrics = {}
        for key in ['body','previous_body','new_body','StartingDirection','CompassDirectionChoice','FirstPersonChoice','Score','FoodPos','FoodPosPrev','FoodPosNext','Dead']:
            self.metrics[key] = {}
        self.StartToCompass_dict()

    def start(self): #this function kicks the game off
        self.snake = Snake([5,self.board["height"] - 5], [5, self.board['width']-5], self.starting_length)
        self.NewBody = self.snake.body
        self.generate_food(start=True)
        if self.gui: self.render_init()
        
    def generate_food(self, start=False): #this creates a new apple somewhere in the board
        food = []
        while food == []:
            food = [randint(1, self.board["height"]), randint(1, self.board["width"])]
            if food in self.NewBody: food = []
        if start:
            self.food = food
            self.food_prev = self.food
        self.food_next = food

    '''The next two functions are specifically about the game visualisation'''

    def render_init(self): #this just initialises curses
        curses.initscr()
        win = curses.newwin(self.board["width"] + 2, self.board["height"] + 2, 0, 0)
        curses.curs_set(0)
        win.nodelay(1)
        win.timeout(200)
        self.win = win #win here is the window
        self.render()

    def render(self): #and then this section actually adds the snake board from the backend to the curses window.
        self.win.clear() 
        self.win.border(0)
        self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ') #adds a score section
        self.win.addch(self.food[0], self.food[1], '@') #adds the apple in the positions [0](y) and [1](x)
        for idx, point in enumerate(self.snake.tail):
            self.win.addch(point[0],point[1],'0')
        self.win.addch(self.snake.head[0],self.snake.head[1],'X')
        self.win.getch()
        

#######
# This is the start of the core engine - the step. It contains multiple functions, defined below.

    def step(self):

        image = CreateOneFrameForChoosingMove(self.snake.body, self.snake.previous_body, self.food, self.food_prev)
        FirstPersonChoice = self.Qmodel.pick_move(image, mode='tiny_epsilon',epsilon=self.epsilon)
        CompassDirectionChoice = self.convertFirstPersontoCompass(FirstPersonChoice)
        self.NewPoint = self.create_new_point(CompassDirectionChoice)

        if self.food_eaten(self.NewPoint):
            self.score = 1
            self.AtLeast1Score = True
            self.generate_food(start=False)
            self.grow()
        else:
            self.move()

        #create logs except for death
        self.CreateLogs(FirstPersonChoice,CompassDirectionChoice)
        #check if dead and log
        self.snake.CheckContigous()
        
        #change settled point of the game
        self.snake.previous_body = self.snake.body.copy()
        self.snake.body = self.NewBody.copy()
        self.snake.direction = CompassDirectionChoice
        self.check_collisions()
        if self.done:
            self.score=-1
            self.metrics['Score'][self.moves] = self.score
        self.CreateDeadLog()
        self.food_prev = self.food.copy()
        self.food = self.food_next.copy()
        self.score = 0
        if self.gui: self.render()
        return self.done

    '''Helper functions for step'''    

    # def model(self):
    #     '''
    #     In: pre-body, body, starting_direction
    #     Out: FirstPersonChoice
    #     '''
    #     FirstPersonChoice = randint(0,2) t
    #     return FirstPersonChoice
    
    #0: forward, 1: left, 2: righ
    # 0 - DOWN, # 1 - RIGHT, # 2 - UP, # 3 - LEFT
    def StartToCompass_dict(self):
        self.IfImFacingDown = {0:0, 1:1, 2:3}
        self.IfImFacingLeft = {0:3, 1:0, 2:2}
        self.IfImFacingRight = {0:1, 1:2, 2:0}
        self.IfImFacingUp = {0:2, 1:3, 2:1}
        self.CompassCoorDict = {0:[1,0],1:[0,1],2:[-1,0],3:[0,-1]}   
    
    def convertFirstPersontoCompass(self, FirstPersonChoice):
        
        if self.snake.direction ==0:
            return self.IfImFacingDown[FirstPersonChoice]
        if self.snake.direction ==1:
            return self.IfImFacingRight[FirstPersonChoice]
        if self.snake.direction ==2:
            return self.IfImFacingUp[FirstPersonChoice]
        if self.snake.direction ==3:
            return self.IfImFacingLeft[FirstPersonChoice]


    def create_new_point(self, CompassDirectionChoice): 
        '''
        In: Compass Direction choice
        Out: the new point that the snake is now at, which is passed to the snake class for him to update his body
        '''
        new_point = self.snake.head.copy()
        added_amount = self.CompassCoorDict[CompassDirectionChoice]
        new_point[0] +=  added_amount[0]
        new_point[1] += added_amount[1]
        return new_point

    def grow(self):
        self.NewBody = [self.NewPoint] + self.snake.body.copy()
        
    def move(self):     
        self.NewBody = ([self.NewPoint] + self.snake.body.copy())
        self.NewBody.pop()
    
    def CreateLogs(self, FirstPersonChoice, CompassDirectionChoice):
        self.metrics['body'][self.moves] = [x for x in self.snake.body[:]]
        self.metrics['previous_body'][self.moves] = [x for x in self.snake.previous_body[:]]
        self.metrics['new_body'][self.moves] = [x for x in self.NewBody[:]]        
        self.metrics['FirstPersonChoice'][self.moves] = FirstPersonChoice
        self.metrics['Score'][self.moves] = self.score
        self.metrics['FoodPos'][self.moves] = self.food
        self.metrics['FoodPosPrev'][self.moves] = self.food_prev
        self.metrics['FoodPosNext'][self.moves] = self.food_next
        # self.metrics['StartingDirection'][self.moves] = int(self.snake.direction)
        # self.metrics['CompassDirectionChoice'][self.moves] = int(CompassDirectionChoice)
        
    def CreateDeadLog(self):
        self.metrics['Dead'][self.moves] = self.done
        self.moves+=1

    def food_eaten(self,NewPoint):
        return NewPoint == self.food #this is an assertion. if the front of the snake == food, add one to food.

    def HitBoardEdge(self):
        if (self.snake.head[0]  in (0,self.board["height"]+1)) or (self.snake.head[1]  in (0,self.board["width"]+1)):
            # self.moves = 100
            return True

    def check_collisions(self):
        #here we just check the very front of the snake
        if self.HitBoardEdge() or self.snake.HasEatenItself():    
            self.done = True


########################################################################
##
##  OK, we're now ready to run the model.
##


if __name__ == "__main__":
    
    Qmodel = SnakeModel()
    Qmodel = torch.load('/users/jacoblourie/RNN_games/Snake/model_checkpoints/MAINSnakemodel6.pt')
    targetModel = SnakeModel(); targetModel.load_state_dict(Qmodel.state_dict())
    
    cycles = 2000
    # save_scores_n = 50

    # epsilon_start = 0.99
    # epsilon_master_reduction = 0.2
    # epsilon_start_floor = 0.25
    # epsilon_floor = 0.01
    epsilon_cycle_reduction = 0.995
 
    record_0_score_every_n = 1000
    record_0_score_every_n_reduction = .999

    plays_per_train_start = 10000    
    frames_per_game_start = 40
    frames_per_game_increase = 0.01

    initial_cycle_section = 100

    performance_metrics = {}
    epsilon = 0.99
    counter = 0
    

    for cycle in range(cycles):
        if cycle >initial_cycle_section:
            frames_per_game = int(frames_per_game*frames_per_game_increase) + 1
            record_0_score_every_n = max(1,int(record_0_score_every_n * record_0_score_every_n_reduction))
        else:
            frames_per_game = frames_per_game_start
        plays_per_train = int(plays_per_train_start + (plays_per_train_start/((cycle+1)**0.2)))
        epsilon_floor = 0.05
        if cycle > 1500:
            epsilon_floor = 0.01
        cycle_score, positive_cycle_score = 0,0
        Metrics = {} #create blank metrics every time.
        for play in range(plays_per_train):

            gui = False

            if play == 0:
                gui = True
            
            game = SnakeGame(Qmodel=Qmodel, targetModel = targetModel, gui = gui, epsilon=epsilon, starting_length=3)
            game.start()
            if gui: game.render()
            done = False
            for i in range(frames_per_game):
                
                done = game.step()
                
                if done:
                    break
            cycle_score += np.sum([x for x in (game.metrics['Score'].values())])
            positive_cycle_score += np.sum([x for x in (game.metrics['Score'].values()) if x>0])
            
            counter+=1
            if (game.AtLeast1Score) or (counter % record_0_score_every_n==0) :
                Metrics[play] = game.metrics

            if gui: 
                curses.endwin()

        
        #save these info
        performance_metrics[cycle] = {}
        performance_metrics[cycle]['epsilon'] = float(epsilon)
        performance_metrics[cycle]['cycle_score'] = int(cycle_score)
        performance_metrics[cycle]['positive_cycle_score'] = int(positive_cycle_score)
        performance_metrics[cycle]['frames_per_game'] = int(frames_per_game)
        performance_metrics[cycle]['plays_per_train'] = int(plays_per_train)
        performance_metrics[cycle]['record_0_scores_n'] = int(record_0_score_every_n)
        
        with open('performance_metrics.json', 'w+') as fp:
                    json.dump(performance_metrics, fp,indent=3)
        epsilon = max(epsilon_floor,epsilon/((cycle+1)**0.0025))
    #############################################################################
    #

        '''Create the manual inputs'''
        batch_size = 32
        sample_percentage = 0.4
        patience = 1
        epochs = 5
        gamma = 0.98
        device = 'cpu' #obsolete reference because i deleted all references to device
        update_target_every_n = 3

        '''Create the model, inputs etc'''
        #create the inputs
        ArrayMax = 250000 #put a max on train size
        AllGamesImage, AllGamesNextImage, FirstPersonMoves, Reward, Done, frames1, frames2, frames3, frames4 = CreateAllXYInputs(Metrics, ArrayMax)
        #take sample
        sample_Image, sample_NextImage, sample_Moves, sample_Reward, sample_Done = get_samples(sample_percentage, AllGamesImage, AllGamesNextImage, FirstPersonMoves, Reward, Done)
        #load into dataloader
        dataset = Dataset(sample_Image, sample_NextImage, sample_Moves, sample_Reward, sample_Done)
        dataset, dataloaders = DatasetsAndDataloaders(dataset, 0.9,0.05, batch_size)
        #save metric and then delete samples in RAM
        performance_metrics[cycle]['dataset_size'] = int(len(sample_Image))
        del AllGamesImage, AllGamesNextImage, FirstPersonMoves, Reward, Done, sample_Image, sample_NextImage, sample_Moves, sample_Reward, sample_Done
        #create optimizer, metrics, loss metric
        optim = torch.optim.Adam(Qmodel.parameters())
        metrics = {}
        metrics['MSE'], metrics['MAE'] = torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()
        criterion = torchmetrics.MeanSquaredError()
        #train model for 2 epochs
        

        '''fit model'''
        fit(dataloaders, Qmodel, targetModel, optim, criterion, metrics, patience, epochs, device, gamma)
        if (cycle+1) % update_target_every_n  == 0:
            targetModel.load_state_dict(Qmodel.state_dict())
        
            
        
        '''Save file every cycle'''
        model_save_name = f'MAINSnakemodel{cycle+1}.pt'
        path = f"/users/jacoblourie/RNN_games/Snake/model_checkpoints/{model_save_name}"
        torch.save(Qmodel, path)
        model_save_name = f'TARGETSnakemodel{cycle+1}.pt'
        path = f"/users/jacoblourie/RNN_games/Snake/model_checkpoints/{model_save_name}"
        torch.save(targetModel, path)
#close the screen
# if gui: 
#     curses.endwin()

# with open('snakeLogs.json', 'w+') as fp:
#     json.dump(Metrics, fp)


'''Old code for randomly making a choice.'''
    # #convert to Compass
        # if self.snake.direction == 0:
        #     CompDirectionChoice = self.FacingUp[FaceDirectionChosen]
        # if self.snake.direction == 1:
        #     CompDirectionChoice = self.FacingRight[FaceDirectionChosen]
        # if self.snake.direction == 2:
        #     CompDirectionChoice = self.FacingDown[FaceDirectionChosen]
        # if self.snake.direction == 3:
        #     CompDirectionChoice = self.FacingLeft[FaceDirectionChosen]
        # return FaceDirectionChosen, CompDirectionChoice