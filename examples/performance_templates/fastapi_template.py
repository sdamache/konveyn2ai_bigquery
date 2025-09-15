"""
FastAPI application module {ID}
Performance test module
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
import logging
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Performance Test API {ID}",
    description="Performance testing module {ID}",
    version="1.{ID}.0"
)

security = HTTPBearer()

class DataModel{ID}(BaseModel):
    id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    score: float = Field(..., ge=0.0, le=100.0)
    is_active: bool = True

class ResponseModel{ID}(BaseModel):
    id: int
    name: str
    email: str
    score: float
    is_active: bool
    processed_at: datetime

@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed for module {ID}")
    response_data = {{
        "message": "Performance Test API {ID}",
        "version": "1.{ID}.0",
        "status": "operational",
        "instance_id": "{ID}"
    }}
    return response_data

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    response_data = {{
        "status": "healthy",
        "module": "{ID}"
    }}
    return response_data

@app.get("/items/{{item_id}}")
async def read_item(
    item_id: int = Path(..., ge=1),
    q: Optional[str] = Query(None, max_length=100)
):
    """Get item by ID"""
    logger.info(f"Fetching item {{item_id}} in module {ID}")

    if item_id > {MAX_ITEM_ID}:
        raise HTTPException(status_code=404, detail="Item not found")

    result = {{
        "item_id": item_id,
        "module": "{ID}",
        "name": f"Item {{item_id}} from module {ID}"
    }}

    if q:
        result["query"] = q

    return result

@app.post("/data/", response_model=ResponseModel{ID})
async def create_data(
    data: DataModel{ID},
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Create new data entry"""
    logger.info(f"Creating data entry: {{data.email}} in module {ID}")

    return ResponseModel{ID}(
        id=data.id + {ID_OFFSET},
        name=data.name,
        email=data.email,
        score=data.score,
        is_active=data.is_active,
        processed_at=datetime.now()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={PORT})