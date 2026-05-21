from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Multi-Agent Coding Assistant API")
app.include_router(router)
