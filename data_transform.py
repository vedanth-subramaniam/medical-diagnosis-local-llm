import os
import csv
import json
import pandas as pd
import numpy as np
from numpy.random import RandomState

def create_jsonl(df: pd.DataFrame, type='train'):
    jsonl_data = []
    
    for _, row in df.iterrows():
        diagnosis = row['label']
        symptoms = row['text']
        prompt = f"You are a medical diagnosis expert. You will give patient symptoms: '{symptoms}'. Question: 'What is the diagnosis I have?'. Response: You may be diagnosed with {diagnosis}."
        jsonl_data.append({"text": prompt})
    
    with open(f'data/{type}.jsonl', 'w') as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry) + '\n')

def transform_data():
    
    df = pd.read_csv('data/medical-diagnosis.csv')
    
    train = df.sample(frac=0.8, random_state=42)
    
    train_set = train.sample(frac=0.8, random_state=42)
    validation_set = train.loc[~train.index.isin(train_set.index)]
    
    test_set = df.loc[~df.index.isin(train.index)]
    
    create_jsonl(train_set, 'train')
    create_jsonl(validation_set, 'validation')
    create_jsonl(test_set, 'test')
            
if __name__ == '__main__':
    transform_data()