# chmod u+x run_pipeline.sh
docker build -t mlscwexample .
docker run --name mlscwexample --rm -v "$PWD":/app mlscwexample python pipeline/run_pipeline.py
s3cmd put File data/testing_set.csv s3://scw-ml-example --acl-public
s3cmd put File data/training_set.csv s3://scw-ml-example --acl-public