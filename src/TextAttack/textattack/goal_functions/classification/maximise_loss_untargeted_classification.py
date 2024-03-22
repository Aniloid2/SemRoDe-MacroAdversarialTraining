from  .classification_goal_function import ClassificationGoalFunction
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
class MaximizeLossGoalFunction(ClassificationGoalFunction):
    """
    A custom goal function that aims to maximize the loss given the input.
    """
    
    def __init__(self,*args, target_min_loss=None,  **kwargs):
        self.loss_fn = CrossEntropyLoss()
        self.target_min_loss = 3 # target_min_loss
        
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, ground_truth_output):
        # Calculate the loss
        # print ('ground trouth',self.ground_truth_output)
        # sys.exit()
        # print ('outputs',model_output,self.ground_truth_output)
        temp_ground_truth_output = torch.tensor(self.ground_truth_output)
        # print ('comaprison',model_output,temp_ground_truth_output)
        loss = self.loss_fn(model_output,temp_ground_truth_output)
        # We achieve our goal if the loss is maximized
        # print ('loss',loss,loss.max(),self.target_min_loss )
        # return loss.max()
    
        
        # if  self.target_min_loss > loss.max():
        if model_output.argmax() != self.ground_truth_output:
            return True
        else:
            return False
        # else:
        #     return True

        temp_ground_truth_output = torch.tensor(self.ground_truth_output)
        print ('comaprison',model_output,temp_ground_truth_output)
        loss = self.loss_fn(model_output,temp_ground_truth_output)

    



    def _get_score(self, model_output, ground_truth_output):
        # Here the score to maximize is just the loss
        temp_ground_truth_output = torch.tensor(self.ground_truth_output)
        score = self.loss_fn(model_output,temp_ground_truth_output)
        
        return score