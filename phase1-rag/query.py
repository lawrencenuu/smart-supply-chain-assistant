from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Test query to retrieve relevant chunks from the vector database
# query = "What deep learning architectures were evaluated in this study, and which one performed best?"

# results = retriever.invoke(query)

# print(f"Retrieved {len(results)} chunks")

# for i, document in enumerate(results, start=1):
#     print(f"\n--- Chunk {i} ---")
#     print(document.page_content)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Testing the LLM with a simple prompt separate from the retrieval process
# response = llm.invoke(
#     "In one sentence, what is Wolffia globosa?"
# )

# print(response.content)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

def answer_question(question):
    response = retrieval_chain.invoke(
        {"input": question}
    )

    # return response["answer"]
    return response

# Testing the answer_question function with a sample question 
# answer = answer_question(
#     "What optimizer and learning rate were used during training?"
# )

# print(answer)

# question = (
#     "What deep learning architectures were evaluated in this study, "
#     "and which one performed best?"
# )
# question = "What dataset was used in this study?"
# question = "What optimizer and learning rate were used during training?"

# response = retrieval_chain.invoke(
#     {"input": question}
# )

# print(response["answer"])