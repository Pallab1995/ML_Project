import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from  sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utilis import save_object

@dataclass
class DataTransformationConfig:
    prepocessor_obj_file_path=os.path.join('artifacts',"processor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is responsible for Data Transformation
        '''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender',
                                 'race_ethnicity',
                                 'parental_level_of_education',
                                 'lunch',
                                 'test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())  # fine for dense arrays
                ]
            )

           

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # safe now because encoder produces dense
               ]
            )

            logging.info(f"Categorical columns:{categorical_columns}")
            logging.info(f"Numerical columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                 ("num_pipeline", num_pipeline, numerical_columns),
                 ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessoring_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            
            logging.info(f"Applying preprocessing object on training and testing dataframe")

            # fit on training features and transform both train and test features
            input_feature_train_arr = preprocessoring_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessoring_obj.transform(input_feature_test_df)

            # concatenate features with targets (ensure matching order / lengths)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr,  np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.prepocessor_obj_file_path,
                obj=preprocessoring_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.prepocessor_obj_file_path,

            )

        except Exception as e:
            raise CustomException(e,sys)

