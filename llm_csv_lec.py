from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import pandas as pd

# # 데이터 파일 경로 설정 (파일 경로는 실제 환경에 맞게 수정해야 합니다)
# data_file_path = './pdfs/23.09.csv'
# # CSV 파일을 데이터 프레임으로 읽어오기
# df = pd.read_csv(data_file_path, encoding='latin1')

# # 데이터 프레임 확인
# print(df.head())  # 데이터 프레임의 처음 몇 행을 출력하여 데이터 구조를 확인합니다.

DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(file_path="./pdfs/2019.csv", encoding='latin1', csv_args={'delimiter': ','})
# loader = CSVLoader(file_path="2019.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
# print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)


query = "What is the mean of all task work time?"

#docs = docsearch.similarity_search(query, k=3)

#print("Result", docs)

# llm = CTransformers(model="/dli/task/llama.cpp/models/Llama-2-7B-GGUF/llama-2-7b.Q4_0.gguf",
#                     temperature=0.1,
#                     model_type="llama",
#                     max_tokens=512,
#                     top_p=1,
#                     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#                     verbose=True, # Verbose is required to pass to the callback manager
#                     )

llm = CTransformers(model="/dli/task/llama.cpp/models/Llama-2-7B-GGUF/llama-2-7b.Q4_0.gguf",
                    temperature=0.1,
                    model_type="llama",
                    max_tokens=512,
                    )

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    #query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])