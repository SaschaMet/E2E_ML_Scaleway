# chmod u+x run_pipeline.sh
git fetch
git reset --hard origin/main

# remove old files
rm Archive.zip

# download the zip file of the dataset
wget http://scw-ml-example.s3.fr-par.scw.cloud/Archive.zip

# move the file to the a new data /directory
mkdir data
mv Archive.zip data/Archive.zip

# unpack the zip file
cd data
unzip -o Archive.zip
cd ..
rm -rf data/__MACOSX

# run the docker image
docker rm mlscwexample
docker build -t mlscwexample .
docker run --name mlscwexample --rm -v "$PWD":/app mlscwexample python pipeline/run_pipeline.py
s3cmd put File data/testing_set.csv s3://scw-ml-example --acl-public
s3cmd put File data/training_set.csv s3://scw-ml-example --acl-public