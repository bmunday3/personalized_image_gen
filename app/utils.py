import streamlit as st
import requests
import json
from mlflow.deployments import get_deploy_client

def make_grid(rows,cols,gap="small"):
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols, gap=gap)
    return grid

def generate_image(endpoint, dataset):
    # Initialize the MLflow deployment client for Databricks
    client = get_deploy_client("databricks")
    
    # Convert the dataset to a dictionary in 'split' orientation
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    
    # Make a prediction request to the specified endpoint with the dataset
    response = client.predict(endpoint=endpoint, inputs=ds_dict)
    
    return response    