import pandas as pd
import numpy as np 

class MealEnvironment:
    def __init__(self, filepath):
        self.data = self._load_data(filepath)
        self.context_features = ['School Name', 'Meal_Type', 'dow'] 
        self.action_column = 'Identifier'
        self.reward_column = 'Served_Total'
        
        self.actions = self.data[self.action_column].unique()
    
        self.action_map = {action: i for i, action in enumerate(self.actions)}
        
        print("Environment initialized successfully!")

    def _load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            print(f"Error: The file at {filepath} was not found.")
            return None

    def get_states(self):
        if self.data is not None:
        
            return self.data[self.context_features]
        return None

    def get_actions(self):
        """
        Returns an array of all unique meal identifiers.
        """
        return self.actionsgit tatu