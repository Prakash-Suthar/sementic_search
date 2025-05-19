from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import sementicsearch_api  

app = FastAPI(
    title="sementic score api",
    description="API for score",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(sementicsearch_api.router, prefix="/api", tags=["STS"])

# Root path
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sementic Score API!"}
