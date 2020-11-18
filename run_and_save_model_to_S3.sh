# chmod u+x run_and_save_model_to_S3.sh
git fetch
git reset --hard origin/main

# remove old files
rm Archive.zip
rm testing_set.csv
rm training_set.csv
rm -rf data

# download the dataset and the train and test splits
wget http://scw-ml-example.s3.fr-par.scw.cloud/Archive.zip
wget http://scw-ml-example.s3.fr-par.scw.cloud/testing_set.csv
wget http://scw-ml-example.s3.fr-par.scw.cloud/training_set.csv

# move the file to the a new data /directory
mkdir data
mv Archive.zip data/Archive.zip
mv training_set.csv data/training_set.csv
mv testing_set.csv data/testing_set.csv

# unpack the zip file
cd data
unzip -o Archive.zip
cd ..
rm -rf data/__MACOSX

# run the docker image
docker rm mlscwexample
docker build -t mlscwexample .
docker run --name mlscwexample --rm -v "$PWD":/app mlscwexample python -m model.model.py
s3cmd put File model/best.model.hdf5 s3://scw-ml-example --acl-public
s3cmd put File model/history.png s3://scw-ml-example --acl-public
s3cmd put File model/my_model.json s3://scw-ml-example --acl-public
