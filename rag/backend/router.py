from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import PromptTemplate
import httpx
from fastapi import FastAPI, WebSocket
import asyncio
from typing import Dict
from typing import List

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
    url = "http://chromadb:8003/db/find_closest/"
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
    url = "http://model:8001/llm/responce/"
    params = {"prompt": prompt}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
            

    if response.status_code == 200:
        return response.json()["ai_response"]
    else:
        response.raise_for_status()


def parse_conversation(conversation):
    formatted_conversation = []

    for index, message in conversation.items():
        role = message['role']
        content = message['content']

        if role == 'user':
            formatted_conversation.append(f"user: {content}")
        elif role == 'assistant':
            formatted_conversation.append(f"ai: {content}")

    return "\n".join(formatted_conversation)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


class ChatRequest(BaseModel):
    text: str
    history: Dict[int, dict]


# Маршрут для добавления текста
@llm_router.post("/chat")
async def chat(request: ChatRequest):
    try:
        formatted_output = parse_conversation(request.history)
        human_input = request.text
        # memory = ConversationBufferMemory()
        context = await get_chromfdb_documents(human_input, 2)
        
        rag_chain = (
            {"context": make_callable(context), 
                # 'history': make_callable(memory.load_memory_variables({})['history']),
                'history': make_callable(formatted_output),
                "human_input": RunnablePassthrough()}
            | prompt
            | get_llm_responce
            | StrOutputParser()
            )
        
        prompt_filled = prompt.format(**{"context": context, 
                                            'history':formatted_output, 
                                            "human_input":human_input})
        
        logging.info(f'PROMT: {prompt_filled}')

    
        chain_responce = await rag_chain.ainvoke(human_input)
        ai_response = chain_responce.replace(prompt_filled, '')

        # memory.save_context({"input": human_input}, {"output": ai_response})

        
        return {"ai_response": f'i got your input {human_input},context: {context}, ai:{ai_response}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# manager = ConnectionManager()


# @llm_router.websocket("/ws/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: str):
#     await websocket.accept()
#     await websocket.send_text(f"Hello, Client {client_id}")
#     while True:
#         data = await websocket.receive_text()
#         await websocket.send_text(f"Message from {client_id}: {data}")
