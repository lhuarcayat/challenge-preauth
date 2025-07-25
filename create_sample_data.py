import pandas as pd
import os

def create_sample_data():

    transaction_1 = {
        'Age': 35,
        'Sex': 'male',
        'Job': 2,  
        'Housing': 'own',
        'Savings account': 'little',
        'Checking account': 'little',
        'Credit amount': 2500,
        'Duration': 12,
        'Purpose': 'car'
    }

    transaction_2 = {
        'Age': 22,
        'Sex': 'male',
        'Job': 1,
        'Housing': 'rent',
        'Savings account': 'none',
        'Checking account': 'none',
        'Credit amount': 15000,
        'Duration': 48,
        'Purpose': 'business'
    }
    
    return transaction_1, transaction_2

def save_sample_data_to_csv(filename='sample_transactions.csv'):
    """
    Guarda los datos de ejemplo en un archivo CSV
    
    Args:
        filename (str): Nombre del archivo CSV
    
    Returns:
        str: Path del archivo guardado
    """

    os.makedirs('inference_data', exist_ok=True)
    filepath = f'inference_data/{filename}'
    
    transaction_1, transaction_2 = create_sample_data()
    df = pd.DataFrame([transaction_1, transaction_2])
    df = df.replace('none',pd.NA)
    df.to_csv(filepath, index=False)

    return filepath

def main():
    
    save_sample_data_to_csv()

if __name__ == "__main__":
    main()