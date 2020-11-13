# chmod u+x run_and_save_model_to_S3.sh
git fetch && git pull
docker build -t mlscwexample .
docker run --name mlscwexample --rm -v "$PWD":/app mlscwexample python -m model.model.py
s3cmd put File model/best.model.hdf5 s3://scw-ml-example --acl-public
s3cmd put File model/history.png s3://scw-ml-example --acl-public
s3cmd put File model/my_model.json s3://scw-ml-example --acl-public
