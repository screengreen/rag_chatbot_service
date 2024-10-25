from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import llmloader




llm_router = APIRouter(
    prefix='/llm',
    tags=['llm']
)

# Модель данных для входного текста
class TextInput(BaseModel):
    text: str


# Маршрут для добавления текста
@llm_router.get("/responce/")
async def chat(prompt):
    try:
        llm = llmloader.get_llm()
        # llm_output = llm(prompt)
        llm_output = 'you ve done it!'

        return {"ai_response": llm_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
