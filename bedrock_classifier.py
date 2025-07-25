import boto3
import pandas as pd
import json
import time
import os

class BedrockRiskClassifier:
    def __init__(self,model_id: str = 'meta.llama3-70b-instruct-v1:0', region: str = 'us-east-1', profile: str = 'analitica'):
        self.model_id = model_id
        self.region = region
        self.profile = profile
        self.bedrock_client = None
    def initialize_bedrock_client(self):
        try:
            session = boto3.Session(
                profile_name= self.profile,
                region_name= self.region
            )
            self.bedrock_client = session.client(service_name='bedrock-runtime')
        except Exception as e:
            print(f'Error al inicializar el cliente: {e}')
    def create_classification_prompt(self, description):
        prompt = f"""You are a senior credit risk analyst with 15 years of experience in banking. 
        Your task is to assess a customer's credit risk based on their profile.

        CUSTOMER PROFILE:
        {description}

        EVALUATION CRITERIA:
        LOW RISK FACTORS (good risk):
        - Age between 25 and 55
        - Skilled or highly skilled employment
        - Own home
        - Established bank accounts (savings/checking)
        - Reasonable credit amounts (<10,000)
        - Short loan term (<24 months)
        - Productive purposes (education, car)

        HIGH RISK FACTORS (bad risk):
        - Age under 25 or over 65
        - Unskilled employment (regardless of residency)
        - Rented housing or no housing
        - No Known bank accounts
        - High credit amounts (>15,000)
        - Long loan duration (>36 months)
        - Risky purposes

        INSTRUCTIONS:
        1. Carefully analyze the customer's profile.
        2. Consider ALL of the factors mentioned.
        3. Assess the credit risk based on the balance between positive and negative factors.
        4. Respond with EXACTLY one of these two words:
        good risk
        bad risk

        Answer:"""

        return prompt

    def call_bedrock_classification(self, prompt):
        try:
            request_body = {
                'prompt':prompt,
                'max_gen_len':50,
                'temperature':0.1,
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
    def classify_descriptions(self,df):
        df_with_targets = df.copy()
        df_with_targets['target'] = ''
        successful_classifications = 0
        failed_classifications = 0
        good_risk_count = 0
        bad_risk_count = 0

        for idx, row in df.iterrows():
            try:
                prompt = self.create_classification_prompt(row['description'])
                classification = self.call_bedrock_classification(prompt)
                print(f'Realizado: {idx}')
                if classification:
                    df_with_targets.at[idx, 'target'] = classification
                    successful_classifications +=1

                    if classification == 'good risk':
                        good_risk_count += 1
                    else:
                        bad_risk_count += 1
                else:
                    df_with_targets.at[idx, 'target'] = 'unknown'
                    failed_classifications += 1
                time.sleep(0.2)
            except Exception as e:
                df_with_targets.at[idx, 'target'] = 'error'
                failed_classifications += 1
        return df_with_targets
    
    def save_classified_data(self, df, output_path: str = 'data/credit_data_with_targets.csv'):
        os.makedirs(os.path.dirname(output_path),exist_ok = True)
        df.to_csv(output_path, index = False)
        return output_path
def main():
    try:
        input_file = 'data/credit_data_with_descriptions.csv'
        output_file = 'data/credit_data_with_targets.csv'
        model_id = 'meta.llama3-70b-instruct-v1:0'
        df = pd.read_csv(input_file)
        classifier = BedrockRiskClassifier(model_id=model_id)
        classifier.initialize_bedrock_client()
        df_with_targets = classifier.classify_descriptions(df)
        classifier.save_classified_data(df_with_targets, output_file)
    except Exception as e:
        print(f'Error en la ejecución: {e}')
if __name__ == "__main__":
    main()
        