# chmod u+x run_and_save_model_to_S3_ci.sh
SERVER_ID="efbcb97d-8678-4847-ae73-b2468b9c4f78"

echo "Starting the Server ..."
scw instance server start -w $SERVER_ID

echo "SSH into server"

# chmod u+x run_pipeline_ci.sh
ssh -i ~/.ssh/scaleway -t root@51.15.112.179 << EOF
	echo "Switching the path to /app"
	cd app
	echo "Run the model"
	chmod u+x run_and_save_model_to_S3.sh && ./run_and_save_model_to_S3.sh
	echo "Done"
EOF

echo "Shutting down the server ..."

scw instance server stop $SERVER_ID