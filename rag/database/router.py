from fastapi import APIRouter, HTTPException

from datatypes import TextInput
from main import manager

db_router = APIRouter(
    prefix='/db',
    tags=['database']
)


# Add new text to database
@db_router.post("/add_text/")
async def add_text(input: TextInput):
    try:
        manager.add_document(input)
        return {"message": "Text added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Маршрут для поиска ближайших векторов и текстов
@db_router.post("/find_closest/")
async def find_closest(input: TextInput, n_results: int = 2):
    try:
        results = manager.get_similar_texts(input, n_results)
        return {'result': results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
