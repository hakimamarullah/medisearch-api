from fastapi import FastAPI
from utils.letor import Ranker
import time
app = FastAPI()

@app.get("/search")
async def search(q: str):
    start_time = time.process_time()
    result = await Ranker.get_documents(query=q) if len(q) != 0 else []
    return {"query_time":time.process_time() - start_time,"message": "success", "data": result}

