from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import EmbModelLoader
import os
from chroma_db import collection



db_router = APIRouter(
    prefix='/llm',
    tags=['llm']
)

# Модель данных для входного текста
class TextInput(BaseModel):
    text: str


# Маршрут для добавления текста
@db_router.post("/get_response/")
async def add_text(input: TextInput):
    try:
        
        return {"message": "Text added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
