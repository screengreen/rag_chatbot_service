from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from router import llm_router

# Define logging rules
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Loading langchain")
    
    yield
    logging.info(f"shutting langchain")

app = FastAPI(lifespan=lifespan)
app.include_router(llm_router)



# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
