from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import PromptTemplate
import httpx
import asyncio

llm_router = APIRouter(
    prefix='/chat',
    tags=['chat']
)

# Модель данных для входного текста
class TextInput(BaseModel):
    text: str

def make_callable(inp):
  return lambda x=None: inp

# Define your custom prompt
template = """
        You are a helpful AI assistant.
        Context:
        {context}

        Dialog history:
        {history}

        Human: {human_input}
        AI:
        """

prompt = PromptTemplate(
    input_variables=["context", "history",  "human_input"],
    template=template
)


async def get_chromadb_responce(text: str, top_k: int = 2):
    url = "http://0.0.0.0:8003/db/find_closest/"
    payload = {
        "text": text,
        "top_k": top_k
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


async def get_chromfdb_documents(text, top_k) -> str:
    result = await get_chromadb_responce(text, top_k )
    result = '\n\n'.join([f'context_{i}: {document}' for i, document in enumerate(result['result']['documents'][0])])

    return result

async def get_llm_responce(prompt):
    url = "http://0.0.0.0:8001/llm/responce/"
    params = {"prompt": prompt}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
            

    if response.status_code == 200:
        return response.json()["ai_response"]
    else:
        response.raise_for_status()




# Маршрут для добавления текста
@llm_router.get("/chat/")
async def chat(human_input):
    try:
        memory = ConversationBufferMemory()
        context = await get_chromfdb_documents(human_input, 2)
        make_callable(memory.load_memory_variables({})['history'])
        
        rag_chain = (
            {"context": make_callable(context), 
                'history': make_callable(memory.load_memory_variables({})['history']),
                "human_input": RunnablePassthrough()}
            | prompt
            | get_llm_responce
            | StrOutputParser()
            )
        
        prompt_filled = prompt.format(**{"context": context, 
                                            'history':memory.load_memory_variables({})['history'], 
                                            "human_input":human_input})
        
        logging.info(f'PROMT: {prompt_filled}')

    
        chain_responce = await rag_chain.ainvoke(human_input)
        if not isinstance(chain_responce, str):
            print(chain_responce)
            print(type(chain_responce))
            raise ValueError("chain_responce is not a string")
        ai_response = chain_responce.replace(prompt_filled, '')

        memory.save_context({"input": human_input}, {"output": ai_response})

        
        return {"ai_response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
