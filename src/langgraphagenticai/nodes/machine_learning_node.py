import pandas as pd
from src.langgraphagenticai.tools.machine_learning_tools import MachineLearningTools

class MachineLearningNode:
    """
    Machine learning tool implementation
    """

    def __init__(self,dataset:pd.DataFrame):
        self.dataset=dataset

    def process(self)->dict:
        """
        Processes the dataset and classify the dataset and also run some basic machine learning models
        Returns:
            dict: type of dataset and stats of basic machine learning models for the give dataset
        """
        machine_learning_tools=MachineLearningTools(self.dataset)
        return { 
            "type_of_dataset":machine_learning_tools.detect_structure(),
            "basic_ml_stats":machine_learning_tools.run_basic_model()
        }