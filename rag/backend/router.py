from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import PromptTemplate


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

# Маршрут для добавления текста
@llm_router.get("/chat/")
async def chat(human_input, context):
    try:

        memory = ConversationBufferMemory()
        #TODO: llm = get_llm()
        make_callable(memory.load_memory_variables({})['history'])
        
        rag_chain = (
            {"context": make_callable(context), 
                'history': make_callable(memory.load_memory_variables({})['history']),
                "human_input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )
        
        prompt_filled = prompt.format(**{"context": context, 
                                            'history':memory.load_memory_variables({})['history'], 
                                            "human_input":human_input})
        
        logging.info(f'PROMT: {prompt_filled}')

    
        chain_responce = rag_chain.invoke(human_input)
        if not isinstance(chain_responce, str):
            raise ValueError("chain_responce is not a string")
        ai_response = chain_responce.replace(prompt_filled, '')

        memory.save_context({"input": human_input}, {"output": ai_response})

        
        return {"ai_response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
