from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from ecommbot.ingest import ingestdata
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_API_KEY=os.getenv("OPEN_API_KEY")

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.If you are getting hi how hello message 
    then just reply it normally dont add extra information. If you get any message that not in your
    database then try to answer it with your own intelligence

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = ChatOpenAI(model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
                 api_key=OPEN_API_KEY,
                 openai_api_base="https://openrouter.ai/api/v1")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("done")
    # chain  = generation(vstore)
    # print(chain.invoke("can you tell me the best bluetooth buds?"))
    
    
    
    