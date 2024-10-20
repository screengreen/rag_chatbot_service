from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import EmbModelLoader
import os
from chroma_db import collection



db_router = APIRouter(
    prefix='/db',
    tags=['database']
)

# Модель данных для входного текста
class TextInput(BaseModel):
    text: str


# Маршрут для добавления текста
@db_router.post("/add_text/")
async def add_text(input: TextInput):
    try:
        collection.upsert(
            documents=[
                input.text
            ],
            ids=[str(hash(input.text))]
        )
        return {"message": "Text added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Маршрут для поиска ближайших векторов и текстов
@db_router.post("/find_closest/")
async def find_closest(input: TextInput, top_k: int = 2):
    try:
        results = collection.query(
                    query_texts=[input.text], # Chroma will embed this for you
                    n_results=top_k # how many results to return
                    )
        return {'result': results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))