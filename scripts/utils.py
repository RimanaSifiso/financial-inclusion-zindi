import pickle
import pandas as pd

def save_model(model, filename:str, save_to='../models'):
    filepath = f'{save_to}/{filename}'
    with open(filepath, 'wb') as pk_file:
        pickle.dump(model, pk_file)


def load_model(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    

def submission(model, preprocessor, filename: str, save_to='../submissions'):
    df_test = pd.read_csv('../data/Test.csv', index_col='uniqueid')
    
    if preprocessor:
        df_test = preprocessor.transform(df_test)
    
    df_test['bank_account'] = model.predict(df_test)
    
    df_test['unique_id'] = df_test.index + ' x ' + df_test['country']
    df_test[['unique_id', 'bank_account']].to_csv(f'{save_to}/{filename}', index=False)
    
