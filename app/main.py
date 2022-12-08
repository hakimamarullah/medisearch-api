from fastapi import FastAPI, status
from utils.letor import Ranker
from fastapi.middleware.cors import CORSMiddleware
from utils.letor import PretrainedModel
from fastapi_pagination import Page, paginate, add_pagination
from model.search_response import SearchResponse
import time
app = FastAPI()
model = None
origins = [
    "http://localhost:3000",
    "https://medisearchengine.netlify.app"
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

@app.get("/search", response_model=Page[SearchResponse], status_code=status.HTTP_200_OK)
async def search(q: str):
    result = await Ranker.get_documents(model, query=q) if len(q) != 0 else []
    return paginate(result)


add_pagination(app)

