# ollama-fastAPI
# git pull > build the image 
docker build -t llm-api .
# delete the old container on docker desktop
# redoply with rthe new image 
docker run -d -p 8288:8288 --name llm-api llm-api