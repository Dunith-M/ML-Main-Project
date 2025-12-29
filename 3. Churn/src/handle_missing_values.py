import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
        self.critical_columns = critical_columns
        logging.info(f'Dropping rows with missing values for columns: {self.critical_columns}')
        
    def handle(self, df):
        df_cleaned = df.dropna(subset=self.critical_columns)
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f"{n_dropped} has been dropped")
        
class Gender(str, Enum):
    MALE = 'Male'
    FEMALE = 'Female'


class GenderPrediction(BaseModel):
    firstname: str
    lastname: str
    pred_gender: Gender
    
class GenderInputer:
    def __init__(self):
        self.groq_client = groq.Groq()

    def _predict_gender(self, firstname, lastname):
        prompt = f"""
            What is the most likely gender (Male or Female) for someone with the first name '{firstname}'
            and last name '{lastname}' ?

            Your response only consists of one word: Male or Female
            """
        response = self.groq_client.chat.completions.create(
                                                            model='llama-3.3-70b-versatile',
                                                            messages=[{"role": "user", "content": prompt}],
                                                            )
        predicted_gender = response.choices[0].message.content.strip()
        prediction = GenderPrediction(firstname=firstname, lastname=lastname, pred_gender=predicted_gender)
        logging.info(f'Predicted gender for {firstname} {lastname}: {prediction}')
        return prediction.pred_gender