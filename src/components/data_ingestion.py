import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTraningConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") #save the traning data in this path
    test_data_path: str=os.path.join('artifacts',"test.csv")  #save the test data
    raw_data_path: str=os.path.join('artifacts',"data.csv")    #for raw data 

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):                      # here will be the data collect query
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')      # data source (change as per data source)
            logging.info('Read the datase as dataFrane')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #if exit then no need to create

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #save the data

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.33, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #save train data
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) #save test data

            logging.info("Ingestion of data Completed ")

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                

            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":  
    obj = DataIngestion()
    # unpack all returned paths (keep raw_path if needed)
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.Initiate_model_trainer(train_arr,test_arr))
	
	











