
# RAG
To run this project, please install Ollama from [here](https://ollama.com/download) (Windows, Linux and MacOS supported)
You will then have to pull the model using:

    ollama pull gemma3:4b
This will pull the default model being used in this project. If you wish to use another model, you can always pull it from the ones listed on Ollama and alter the model parameter of the generate_answer() method
##
Finally, install the dependencies using:

    pip install -r requirements.txt
## 
Now you can place your documents in the pdfs folder and run the main.py file
