# OpenAI-to-Open-source
A simple RAG POC that shows how to use vLLM to move from OpenAI GPT-4 LLM to IBM Granite instruct model.This example uses Chroma as the database for storing embeddings of a PDF file and LangChain as the framework for Retrieval-Augmented Generation (RAG).

1. Install Python 3.8 or higher
2. Download this code repository (install git if it is not already setup, You can also downlod the zip file directly from the main page under code option as an alternate)

```
      brew install git
      git clone https://github.com/AhilanPonnusamy/OpenAI-to-Open-source.git
```
   
4. Create a new virtual environment from a terminal
   
```
     python3 -m venv .venv
```

5. Activate the vitual environment

```
    source .venv/bin/activate
```


6. Install dependencies

```
    pip install -r requirements.txt
```

7. Add your OpenAI API Key to **OPENAI_API_KEY** variable in **.env** file. You can create OpenAI API Key at **https://platform.openai.com/api-keys**

## Testing OpenAI GPT-4 LLM and OpenAI Embeddings 

1. Run OpenAI chatbot App

```
    streamlit run chatbot_ui.py
```

2. Try a random question **what is the origin of ML?**. Once submitted, you will see some activity in streamlit console and in about 20 seconds a generic LLM response is dislayed in the UI as shown below.
![App UI](./images/RandomWithoutRAG.png)

3. Upload a PDF file (you may also try MLbasics.pdf provided in this project). Wait for the file to be uploaded and embeddings created and stored in the vector DB. Try a random question with the information provided in the uploaded file ** e.g., **what is a loss function?**. You will see a relevant response displayed on the screen as shown below.
![App UI](./images/GPT4-with-RAG.png)
   
## Testing with Open source IBM Granite and all-MiniLM-L6-v2 Embeddings 

1. **chatbot_ui_granite.py** contains all changes required for IBM Granite and all-MiniLM-L6-v2 Embedding model integration.

2. Setup InstructLab following the instruction up to **Initializing InstructLab and a Taxonomy project** section from **https://developers.redhat.com/blog/2024/06/12/getting-started-instructlab-generative-ai-model-tuning#**.  
3. Download Granite-7b-instruct GGUF model from **https://huggingface.co/QuantFactory/granite-7b-instruct-GGUF/tree/main** and move it to models folders. For this POC I used **granite-7b-instruct.Q4_K_M.gguf** model.
4. Update the **model_path** under **serve** in **config.yaml** to use the downloaded granite model as shown below
   ```
      model_path: models/granite-7b-instruct.Q4_K_M.gguf
   ```
5. Serve the model
   ```
      ilab model serve
   ```
6. In a new terminal start the RAG demo with Granite model in the virtual environment.
   ```
       streamlit run chatbot_ui_granite.py
   ``` 
7. 
8. 
9. 
10. 
11.  

***Have fun!!!!!***
