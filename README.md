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

4. Try various prompts from the main folder README file. You will see the responses are much more aligned with the context with less hallucination and warning messages.
>![App UI](../images/Finetuned-output.png)  
   
***Have fun!!!!!***
