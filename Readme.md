# Machine Learning in the Cloud
An End-To-End Project from Data Preparation to Model Deployment

## Description

## Prerequisites
- A Scaleway Account: https://console.scaleway.com/register
- Scaleway CLI: https://www.scaleway.com/en/docs/manage-cloud-servers-with-scaleway-cli/
- Configured SSH Key: https://www.scaleway.com/en/docs/configure-new-ssh-key/
- Optional: Configure s3cmd for file uploading https://www.scaleway.com/en/docs/object-storage-with-s3cmd/


traing the model with: python -m model.model.py


## Todo:
- [x] Get csv with metadata
- [x] Create pipeline
- [x] Store the csv under /data
- [x] Create an EDA Notebook
- [x] Create an Image Data Generator under /model (train and test)
- [x] Create a Model under /model
- [x] Save the Model under /model
- [] Create a .sh script for automatically saving the model to S3
- [] Create a Dockerfile to automatically run the pipeline, train the model and execute the .sh script
- [] Create an Inference Notebook?
- [] Create Tensorboard?
- [] https://cml.dev/ + https://dvc.org/ + https://github.com/EthicalML/awesome-production-machine-learning