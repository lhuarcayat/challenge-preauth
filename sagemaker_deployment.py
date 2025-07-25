import boto3
import json
import joblib
import os
from datetime import datetime
import sagemaker
from sagemaker.xgboost import XGBoostModel
from sagemaker import image_uris

class SageMakerModelDeployment:
    def __init__(self, role_arn, region="us-east-1", profile_name='analitica'):
        self.role_arn = role_arn
        self.region = region
        self.boto_session = boto3.Session(profile_name=profile_name, region_name=region)
        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.endpoint_name = None
        
    def load_training_info(self, training_info_path):
        with open(f"{training_info_path}/training_info.json", 'r') as f:
            self.training_info = json.load(f)

        self.encoders = joblib.load(f"{training_info_path}/encoders.pkl")
        
        print(f"Información de entrenamiento cargada:")
        print(f"- Job Name: {self.training_info['job_name']}")
        print(f"- Model Path: {self.training_info['output_path']}")
        print(f"- Duración: {self.training_info['duration']}")
        
    def get_model_path(self):

        if self.training_info['use_tuning']:
            best_job = self.training_info['best_training_job']
            model_path = f"{self.training_info['output_path']}/{best_job}/output/model.tar.gz"
        else:
            model_path = f"{self.training_info['output_path']}/model.tar.gz"
        
        return model_path
    
    def create_model(self):

        model_path = self.get_model_path()

        image_uri = image_uris.retrieve(
            framework="xgboost",
            region=self.region,
            version="1.5-1",
            py_version="py3"
        )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"fraud-detection-model-{timestamp}"
        
        self.xgb_model = XGBoostModel(
            model_data=model_path,
            role=self.role_arn,
            image_uri=image_uri,
            framework_version="1.5-1",
            py_version="py3",
            sagemaker_session=self.sagemaker_session,
            name=model_name
        )
        
        print(f"Modelo creado: {model_name}")        
        return self.xgb_model
    
    def deploy_endpoint(self, instance_type='ml.t2.medium', initial_instance_count=1):

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.endpoint_name = f"fraud-detection-endpoint-{timestamp}"
        
        print(f"Desplegando endpoint: {self.endpoint_name}")

        self.predictor = self.xgb_model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=self.endpoint_name,
            serializer=sagemaker.serializers.CSVSerializer(),
            deserializer=sagemaker.deserializers.CSVDeserializer()
        )
        
        print(f"Endpoint desplegado: {self.endpoint_name}")
        return self.endpoint_name
    
    def save_deployment_info(self):

        deployment_info = {
            'endpoint_name': self.endpoint_name,
            'model_name': self.xgb_model.name,
            'training_job': self.training_info['job_name'],
            'deployment_time': datetime.now().isoformat(),
            'region': self.region
        }
        
        os.makedirs('deployment_info', exist_ok=True)
        
        with open(f"deployment_info/{self.endpoint_name}.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"Información de deployment guardada en: deployment_info/{self.endpoint_name}.json")
        
    def delete_endpoint(self, endpoint_name=None):
        if endpoint_name is None:
            endpoint_name = self.endpoint_name
            
        if endpoint_name:
            try:
                self.predictor.delete_endpoint()
                print(f"Endpoint {endpoint_name} eliminado")
            except Exception as e:
                print(f"Error eliminando endpoint: {e}")
        else:
            print("No hay endpoint activo para eliminar")

def main():
    ROLE_ARN = 'arn:aws:iam::442431377530:role/challenge-preauth-role'
    TRAINING_INFO_PATH = "train_info/ai-fraud-20250724-183602"  

    try:
        deployer = SageMakerModelDeployment(role_arn=ROLE_ARN)
        deployer.load_training_info(TRAINING_INFO_PATH)
        deployer.create_model()
        endpoint_name = deployer.deploy_endpoint(
            instance_type='ml.t2.medium',  # Instancia económica para pruebas
            initial_instance_count=1
        )
        deployer.save_deployment_info()
        
        print(f"Deployment completado")
        print(f"Endpoint: {endpoint_name}")
        print(f"Región: {deployer.region}")

    except Exception as e:
        print(f"Error durante el deployment: {e}")
        

if __name__ == "__main__":
    main()