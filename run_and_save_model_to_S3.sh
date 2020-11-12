# chmod u+x run_and_save_model_to_S3.sh
docker build -t mlscwexample .
docker run --name mlscwexample --rm -v "$PWD":/app mlscwexample python -m model.model.py
s3cmd put File model/best.model.hdf5 s3://ml-scaleway-example/ --acl-public
s3cmd put File model/history.png s3://ml-scaleway-example/ --acl-public
s3cmd put File model/my_model.json s3://ml-scaleway-example/ --acl-public
