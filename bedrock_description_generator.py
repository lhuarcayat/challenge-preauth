import boto3
import pandas as pd
import json
import time
import os

#En mi caso tengo los permisos en un 'profile' por lo que usaré las credenciales de ese perfil
#Si tienes las credenciales de un usuario IAM simple entonces el código se tendrá que modificar un poco

class BedrockDescriptionGenerator:
    def __init__(self,model_id: str = 'meta.llama3-70b-instruct-v1:0', region: str = 'us-east-1', profile: str = 'analitica'):
        self.model_id = model_id
        self.region = region
        self.profile = profile
        self.bedrock_client = None
        self.job_mapping = {
            0: "unskilled and non-resident",
            1: "unskilled and resident", 
            2: "skilled",
            3: "highly skilled"
        }
    
    def initialize_bedrock_client(self):
        try:
            session = boto3.Session(
                profile_name= self.profile,
                region_name= self.region
            )
            self.bedrock_client = session.client(service_name='bedrock-runtime')
        except Exception as e:
            print(f'Error al inicializar el cliente: {e}')
    
    def create_description_prompt(self, row: pd.Series):
        job_description = self.job_mapping.get(row['Job'])
        savings_account = row['Savings account'] if row['Savings account'] != 'unknown' else 'no information about savings account'
        checking_account = row['Checking account'] if row['Checking account'] != 'unknown' else 'no information about checking account'

        prompt = f"""You are an expert financial analyst. Your task is to create a professional and concise description of a bank customer based on the following structured data.
        
        CUSTOMER DATA:
        - Age: {row['Age']} years old
        - Sex: {row['Sex']}
        - Job: {job_description}
        - Housing: {row['Housing']}
        - Savings account: {savings_account}
        - Checking account: {checking_account}
        - Credit amount: s/.{row['Credit amount']:,} (Peruvian soles)
        - Duration: {row['Duration']} months
        - Purpose: {row['Purpose']}

        INSTRUCTIONS:
        1. Create a narrative description in English consisting of 2-3 sentences.
        2. Include ALL the provided information in a natural and readable flow.
        3. Use a professional but understandable tone.
        4. Explicitly mention when information is not available.
        5. Do not add interpretations or value judgments.

        OUTPUT:
        Returns only the description without headers or additional explanations. 
        
        DESCRIPTION:"""

        return prompt
    
    def call_bedrock_api(self, prompt):
        try:
            #parámetros para LLama3
            request_body = {
                'prompt':prompt,
                'max_gen_len':300,
                'temperature':0.3,
                'top_p':0.9
            }
            response = self.bedrock_client.invoke_model(
                modelId = self.model_id,
                body = json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            description = response_body['generation'].strip()

            return description
        except Exception as e:
            print(f'Error al inicializar el cliente: {e}')
            return None
        
    def generate_descriptions(self,df):
        df_with_descriptions = df.copy()
        df_with_descriptions['description'] = ''
        successful_generations = 0
        failed_generations = 0

        for idx, row in df.iterrows():
            try:
                prompt = self.create_description_prompt(row)
                description = self.call_bedrock_api(prompt)
                print(f'Realizado: {idx}')
                if description:
                    df_with_descriptions.at[idx,'description'] = description
                    successful_generations += 1
                else:
                    df_with_descriptions.at[idx,'description'] = 'Error: No se pudo generar descripcion'
                    failed_generations += 1
                time.sleep(0.2)
            except Exception as e:
                df_with_descriptions.at[idx, 'description'] = f'Error: {str(e)}'
                failed_generations += 1
        return df_with_descriptions
    
    def save_descriptions(self, df, output_path: str = 'data/credit_data_with_descriptions.csv'):
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        df.to_csv(output_path, index = False)
        return output_path

def main():
    try:
        input_file = 'data/processed_credit_data.csv'
        output_file = 'data/credit_data_with_descriptions.csv'
        model_id = 'meta.llama3-70b-instruct-v1:0'
        df = pd.read_csv(input_file)
        generator = BedrockDescriptionGenerator(model_id=model_id)
        generator.initialize_bedrock_client()
        df_with_descriptions = generator.generate_descriptions(df)
        generator.save_descriptions(df_with_descriptions,output_file)
    except Exception as e:
        print(f'Error en la ejecución: {e}')

if __name__ == '__main__':
    main()







    

