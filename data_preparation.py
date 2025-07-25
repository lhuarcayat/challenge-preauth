import pandas as pd
import os

class DataPreparator:
    def __init__(self,file_path:str):
        self.file_path = file_path
        self.df = None
        self.original_shape = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, delimiter = ';')
            self.original_shape = self.df.shape
            print(f'Dataset cargado: {self.original_shape[0]}')
            return self.df
        except Exception as e:
            print(f'Error al cargar el dataset: {e}')

    def handle_missing_values(self):
        columns = ['Savings account','Checking account']
        for column in self.df.columns:
            self.df[column] = self.df[column].fillna('unknown')
        return self.df
    
    def save_processed_data(self,output_path: str = 'data/processed_credit_data.csv'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index = False)
        print('Dataset procesado')
        return output_path

def main():
    input_file = 'data/credir_risk_reto.csv'
    output_file = 'data/processed_credit_data.csv'

    try:
        preparator = DataPreparator(input_file)
        preparator.load_data()
        preparator.handle_missing_values()
        preparator.save_processed_data(output_file)
    except Exception as e:
        print(f'Error en la preparación: {e}')

if __name__ == "__main__":
    main()

