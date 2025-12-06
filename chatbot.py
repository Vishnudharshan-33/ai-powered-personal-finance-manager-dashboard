from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate
import faiss
import numpy as np 

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# Load financial documents
loader = TextLoader("./finance_analysis_report.txt")
docs = loader.load()
text = docs[0].page_content

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents([text])
chunks_texts = [doc.page_content for doc in chunks]

# create embeddings
embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(chunks_texts)

#Build Search Index
dimension = len(embeddings[0])
vectors = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(dimension) 
index.add(vectors)


llm = ChatOpenAI(model= "gpt-4o-mini")


def get_answer(question):
    """Answer user questions based on financial documents."""

    # Find relevant chunks
    query_vector = np.array(embeddings_model.embed_query(question)).reshape(1,-1)
    distances, indices = index.search(query_vector, k=3)

    context = "\n\n".join([chunks_texts[idx] for idx in indices[0]])

    # Create prompt
    prompt = f"""Based on this financial data : 
{context}

Question: {question}

Answer concisely and accurately based only on the data provided."""
    
    #Get response from LLM
    response = llm.invoke(prompt)
    
    # Extract text content from AIMessage object
    return response.content

# Web interface
@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat',methods=['POST'])
def chat():
    question = request.form['user_input']
    answer = get_answer(question)
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)



