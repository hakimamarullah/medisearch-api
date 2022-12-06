from fastapi import FastAPI, status
from utils.letor import Ranker
from fastapi.middleware.cors import CORSMiddleware
from utils.letor import PretrainedModel
import time
app = FastAPI()
model = None
origins = [
    "http://localhost"
]

@app.on_event('startup')
async def startup():
    global model
    model = PretrainedModel()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "code": status.HTTP_200_OK
    }

@app.get("/search", status_code=status.HTTP_200_OK)
async def search(q: str):
    start_time = time.process_time()
    result = await Ranker.get_documents(model, query=q) if len(q) != 0 else []
    return {"query_time":time.process_time() - start_time,"message": "success", "data": result}

