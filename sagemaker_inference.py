import boto3
import pandas as pd
import joblib
from datetime import datetime
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
import os
import json

class FraudDetectionInference:
    def __init__(self, region="us-east-1", profile_name='analitica'):
        self.region = region
        self.boto_session = boto3.Session(profile_name=profile_name, region_name=region)
        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.encoders = None
        self.predictor = None
        self.feature_columns = ['Age', 'Sex', 'Job', 'Housing', 'Savings account', 
                               'Checking account', 'Credit amount', 'Duration', 'Purpose']
        
    def load_encoders(self, training_info_path):
        encoder_path = f"{training_info_path}/encoders.pkl"

        self.encoders = joblib.load(encoder_path)
        print(f"Encoders cargados desde: {encoder_path}")
        
    def connect_to_endpoint(self, endpoint_name):
        try:
            self.predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=self.sagemaker_session,
                serializer=CSVSerializer(),
                deserializer=CSVDeserializer()
            )
            print(f"Conectado al endpoint: {endpoint_name}")
        except Exception as e:
            print(f"Error conectando al endpoint {endpoint_name}: {e}")
    
    def preprocess_data(self, data):
        df = pd.DataFrame([data])
        
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Faltan las siguientes columnas: {missing_cols}")
        
        df = df[self.feature_columns].copy()
        
        categorical_columns = ['Sex', 'Housing', 'Savings account', 'Checking account', 'Purpose']
        account_columns = ['Savings account', 'Checking account']
        for col in account_columns:
                if col in self.encoders:
                    original_value = df[col].iloc[0]
                    if pd.isna(original_value) or original_value == 'none' or original_value == '':
                        df[col] = 'unknown'

        for col in categorical_columns:
                if col in self.encoders:
                    try:
                        df[col] = self.encoders[col].transform(df[col].astype(str))
                    except ValueError as e:
                        print(f"Error codificando {col}: {e}")
                        df[col] = 0
                else:
                    raise ValueError(f"No se encontró encoder para la columna: {col}")
            
        return df
        
    def predict_single(self, data):
        if self.predictor is None:
            raise ValueError("Primero debe conectarse a un endpoint")
        processed_data = self.preprocess_data(data)
        csv_input = ','.join(map(str, processed_data.iloc[0].values))
        prediction = self.predictor.predict(csv_input)

        prob_good_risk = float(prediction[0][0])
        prob_fraud = 1 - prob_good_risk 

        is_fraud = prob_fraud > 0.5
        
        predicted_class = 0 if is_fraud else 1

        if 'target' in self.encoders:
            try:
                target_decoded = self.encoders['target'].inverse_transform([predicted_class])[0]
            except Exception as e:
                print(f"Error decodificando target: {e}")
                target_decoded = "bad risk" if is_fraud else "good risk"

        result = {
            'input_data': data,
            'probability_fraud': prob_fraud,
            'is_fraud': is_fraud,
            'target': target_decoded,
            'confidence': max(prob_fraud, 1 - prob_fraud),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
def load_and_test_csv(fraud_detector, csv_path='inference_data/sample_transactions.csv'):
       
    df = pd.read_csv(csv_path)
    results = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        transaction_data = row.to_dict()
        for key, value in transaction_data.items():
            if pd.isna(value):
                transaction_data[key] = None
            
        result = fraud_detector.predict_single(transaction_data)
        results.append(result)

    good_risk = sum(1 for r in results if r['target'] == 'good risk')
    bad_risk = sum(1 for r in results if r['target'] == 'bad risk')
    
    print(f"Good risk: {good_risk}")
    print(f"Bad risk: {bad_risk}")

    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"results/inference_results_{timestamp}.json"
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados guardados en: {json_filename}")
    return results

def main():
    ENDPOINT_NAME = "fraud-detection-endpoint-20250724-200924"
    TRAINING_INFO_PATH = "train_info/ai-fraud-20250724-183602"

    try:
        csv_file = 'inference_data/sample_transactions.csv'
        fraud_detector = FraudDetectionInference()
        fraud_detector.load_encoders(TRAINING_INFO_PATH)
        fraud_detector.connect_to_endpoint(ENDPOINT_NAME)
        load_and_test_csv(fraud_detector, csv_file)
    except Exception as e:
        print(f"Error durante la inferencia:{e}")

if __name__ == "__main__":
    main()