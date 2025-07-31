import pandas as pd
from src.langgraphagenticai.ML.ml_function import MachineLearningFunction

class MachineLearningTool:
    """
    Machine learning tool implementation
    """

    def __init__(self,dataset:pd.DataFrame):
        self.dataset=dataset

    def process(self)->str:
        """
        Processes the dataset and classify the dataset and also run some basic machine learning models
        Returns:
            str: type of dataset and best machine learning models for the give dataset
        """
        machine_learning_function=MachineLearningFunction(self.dataset)
        
        type_of_dataset=machine_learning_function.detect_structure()
        best_model=machine_learning_function.run_basic_model()

        return f"The provided data is {type_of_dataset} dataset and the best fit model is {best_model}"