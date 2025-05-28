This is a question answering chatbot and it will answer based on the data uploaded in the pdf file.

Upload a PDF and ask questions about it
- Uses local LLMs (LLaMA 3.2, Phi 3.5, Gemma 2B, Qwen 7B, etc.)
- Local vector search with semantic embeddings
- Completely offline — runs on your local machine
- Simple and elegant UI built with **Streamlit**

This project runs on local machine using llm and Ollama.
Since it uses local machine to run llm the performance of this project depends upon the hardware of the system.
It is recommended to have a NIVIDEA graphics on your machine.


steps to run the project
1.DOWNLOAD PYTHON PACKAGES
(paste these code in the terminal)

pip install sentence-transformers
pip install langchain-ollama
pip install chromadb
pip install streamlit
pip install PyPDF2
pip install langchain

2.INSTALL ollama locally on your system 
(by visiting ollama website and downloading and setting up the installer)
To check whether ollama is installed or not run this on terminal "ollama"

3.INSTALL models on ollama
(open terminal and paste the commands after ollama is installed)

ollama pull llama3.2       
ollama pull phi3.5          
ollama pull gemma2:2b      
ollama pull qwen2.5:7b       


4.Save the python file
  Start Ollama using "ollama serve"

  
5.To run the program open terminal in that python file folder and run this command
python -m streamlit run <app.py/optimizedapp.py>

Now follow the link in the terminal output to the browser.


