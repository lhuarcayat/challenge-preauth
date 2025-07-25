import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.inputs import TrainingInput
import os
import json
import joblib
from datetime import datetime

class SageMakerModelTrainer:
    def __init__(self, role_arn, bucket_name, region: str = "us-east-1",profile_name: str = 'analitica'):
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region = region
        self.boto_session = boto3.Session(profile_name=profile_name, region_name = region )
        self.sagemaker_session = sagemaker.Session(boto_session = self.boto_session)
        self.s3_prefix = 'challenge-ai-fraud-detection'
        self.s3_urls = {}
        self.scale_pos_weight = None

    def prepare_data_training(self, df):
        feature_columns = ['Age', 'Sex', 'Job', 'Housing', 'Savings account', 
                          'Checking account', 'Credit amount', 'Duration', 'Purpose']
        X = df[feature_columns].copy()
        y = df['target'].copy()
        #enconding
        encoders = {}
        categorical_columns = ['Sex','Housing','Savings account','Checking account', 'Purpose'] #Job ya está codificado
        for col in categorical_columns:
            label_encoder = LabelEncoder()
            X[col] = label_encoder.fit_transform(X[col].astype(str))
            encoders[col] = label_encoder
        
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        encoders['target'] = target_encoder
        #entrenamiento (stratify asegura que la proporción de clases se a mantenga en ambos conjuntos)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        good_count = np.sum(y_train == 0)  
        bad_count = np.sum(y_train == 1)   
        if good_count > bad_count:
            scale_pos_weight = good_count / bad_count
        else:
            scale_pos_weight = bad_count / good_count
        return X_train, X_test, y_train, y_test, encoders, scale_pos_weight
    
    def upload_data_to_s3(self, X_train, X_test, y_train, y_test):
        #formato para XGBOOST, el target va primero
        train_data = pd.concat([pd.Series(y_train), X_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([pd.Series(y_test), X_test.reset_index(drop=True)], axis=1)
        
        #guardar data en local
        os.makedirs('data', exist_ok=True)
        train_file = "train_data.csv"
        test_file = "test_data.csv"

        train_data.to_csv(f'data/{train_file}', header = False, index = False)
        test_data.to_csv(f'data/{test_file}', header = False, index = False)

        #subir a s3
        s3_client = self.boto_session.client('s3', region_name=self.region)

        train_s3_key = f"{self.s3_prefix}/data/train/{train_file}"
        test_s3_key = f"{self.s3_prefix}/data/test/{test_file}"

        s3_client.upload_file(f'data/{train_file}', self.bucket_name, train_s3_key)
        s3_client.upload_file(f'data/{test_file}', self.bucket_name, test_s3_key)

        s3_urls = {
            'train':f's3://{self.bucket_name}/{train_s3_key}',
            'test':f's3://{self.bucket_name}/{test_s3_key}'
        }

        return s3_urls
    
    def create_xgboost_estimator(self, output_path, scale_pos_weight):
        image_uri = image_uris.retrieve(
            framework="xgboost",
            region=self.region,
            version="1.5-1",
            py_version="py3"
        )       
    
        xgb_estimator = Estimator(
        image_uri = image_uri,
        instance_type='ml.m5.xlarge', 
        instance_count=1,
        output_path=output_path,
        role=self.role_arn,
        sagemaker_session=self.sagemaker_session,
        #hiperparámetros base, se optimizará después
        hyperparameters={
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'num_round': 100,
            'scale_pos_weight':scale_pos_weight,
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'verbosity': 1
            }
        )
        return xgb_estimator
    
    def setup_hyperparameter_tuning(self, estimator, s3_urls):
        hyperparameter_ranges = {
            'max_depth': IntegerParameter(3, 10),
            'eta': ContinuousParameter(0.01, 0.3),
            'subsample': ContinuousParameter(0.6, 1.0),
            'colsample_bytree': ContinuousParameter(0.6, 1.0),
            'min_child_weight': IntegerParameter(1, 10)
        }
        
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name='validation:auc',
            objective_type='Maximize',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=10,  
            max_parallel_jobs=2, 
            strategy='Bayesian'
        ) 
        return tuner

    def train_model(self, use_hyperparameter_tuning):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"ai-fraud-{timestamp}"
        
        output_path = f"s3://{self.bucket_name}/{self.s3_prefix}/models/{job_name}"
        
        try:
            estimator = self.create_xgboost_estimator(output_path,self.scale_pos_weight)
            
            train_input = TrainingInput(self.s3_urls['train'], content_type='text/csv')
            test_input = TrainingInput(self.s3_urls['test'], content_type='text/csv')
            
            training_info = {
                'job_name': job_name,
                'start_time': datetime.now(),
                'output_path': output_path,
                'use_tuning': use_hyperparameter_tuning
            }
            
            if use_hyperparameter_tuning:
                tuner = self.setup_hyperparameter_tuning(estimator, self.s3_urls)

                tuner.fit({
                    'train': train_input,
                    'validation': test_input
                }, job_name=job_name)
                
                training_info['tuner'] = tuner
                training_info['tuning_job_name'] = tuner.latest_tuning_job.job_name                   
            else:
                estimator.fit({
                    'train': train_input,
                    'validation': test_input
                }, job_name=job_name)
                
                training_info['estimator'] = estimator
                training_info['training_job_name'] = estimator.latest_training_job.job_name  
            return training_info
            
        except Exception as e:
            print(f'Error en la implementación {e}')
    
    def wait_training_completion(self, training_info):
        try:
            if training_info['use_tuning']:
                tuner = training_info['tuner']
                tuner.wait()
            
                best_training_job = tuner.best_training_job()
                training_info['best_training_job'] = best_training_job
                training_info['best_estimator'] = tuner.best_estimator()
            
            else:
                estimator = training_info['estimator']
                estimator.wait()
            training_info['end_time'] = datetime.now()
            training_info['duration'] = training_info['end_time'] - training_info['start_time']
            return training_info
        except Exception as e:
            print(f'Error en la espera del entrenamiento: {e}')
    
    def save_training_info(self, training_info, encoders):
        train_info_dir = f"train_info/{training_info['job_name']}"
        os.makedirs(train_info_dir, exist_ok=True)
        
        joblib.dump(encoders, f"{train_info_dir}/encoders.pkl")

        training_summary = {
            'job_name': training_info['job_name'],
            'output_path': training_info['output_path'],
            'duration': str(training_info['duration']),
            'use_tuning': training_info['use_tuning']
        }
        
        if training_info['use_tuning']:
            training_summary['tuning_job_name'] = training_info['tuning_job_name']
            training_summary['best_training_job'] = training_info['best_training_job']

        with open(f"{train_info_dir}/training_info.json", 'w') as f:
            json.dump(training_summary, f, indent=2)

        return train_info_dir

def main():
    try:
        ROLE_ARN = 'arn:aws:iam::442431377530:role/challenge-preauth-role'
        BUCKET_NAME = 'challenge-preauth'
        input_file = 'data/credit_data_with_targets.csv'
        df = pd.read_csv(input_file)
        trainer = SageMakerModelTrainer(
            role_arn=ROLE_ARN,
            bucket_name= BUCKET_NAME
        )

        X_train, X_test, y_train, y_test, encoders, scale_pos_weight = trainer.prepare_data_training(df)
        trainer.scale_pos_weight = scale_pos_weight
        trainer.s3_urls = trainer.upload_data_to_s3(X_train, X_test, y_train, y_test)
        training_info = trainer.train_model(use_hyperparameter_tuning=True)
        final_training_info = trainer.wait_training_completion(training_info)
        trainer.save_training_info(final_training_info, encoders)

    except Exception as e:
        print(f'Error en entrenamiento: {e}')

if __name__ == '__main__':
    main()