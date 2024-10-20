# OpenAI-to-Open-source
A simple RAG POC that shows how to use vLLM to move from OpenAI GPT-4 LLM to IBM Granite instruct model. 

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

4. Activate the vitual environment

```
    source .venv/bin/activate
```


5. Install dependencies

```
    pip install -r requirements.txt
```

6. Add your OpenAI API Key to **OPENAI_API_KEY** variable in **.env** file. You can create OpenAI API Key at **https://platform.openai.com/api-keys**
   
7. Run OpenAI chatbot App

```
    streamlit run chatbot_ui.py
```
   
***Have fun!!!!!***
