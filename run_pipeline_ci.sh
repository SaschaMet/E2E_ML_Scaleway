# chmod u+x run_pipeline_ci.sh
SERVER_ID="1307e8c9-be76-4204-ac7b-76b6c0fdb5f2"

echo "Starting the Server ..."
scw instance server start -w $SERVER_ID

echo "SSH into server"

# chmod u+x run_pipeline_ci.sh
ssh -i ~/.ssh/scaleway -t root@51.158.188.227 << EOF
	echo "Switching the path to /app"
	cd app
	echo "Run the pipeline"
	chmod u+x run_pipeline.sh && ./run_pipeline.sh
	echo "Done"
EOF

echo "Shutting down the server ..."

scw instance server stop $SERVER_ID