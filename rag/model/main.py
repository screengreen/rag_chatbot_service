from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from router import llm_router
from models import llmloader
import uvicorn



# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Loading llm")
    llmloader.load_generation_pipeline()
    
    yield
    logging.info(f"shutting llm")

app = FastAPI(lifespan=lifespan)
app.include_router(llm_router)



# Запуск приложения
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
