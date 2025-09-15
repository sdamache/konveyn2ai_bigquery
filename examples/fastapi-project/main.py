"""
Sample FastAPI application for testing M1 ingestion
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Sample API",
    description="A sample FastAPI application for testing M1 ingestion",
    version="1.0.0",
)

class User(BaseModel):
    id: int
    name: str
    email: str
    active: bool = True

class UserCreate(BaseModel):
    name: str
    email: str

# In-memory storage for demo
users_db = []

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Hello World", "service": "sample-api"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/users", response_model=List[User])
async def get_users(skip: int = 0, limit: int = 10):
    """Get all users with pagination"""
    return users_db[skip : skip + limit]

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    """Get a specific user by ID"""
    for user in users_db:
        if user["id"] == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

@app.post("/users", response_model=User)
async def create_user(user: UserCreate):
    """Create a new user"""
    new_user = {
        "id": len(users_db) + 1,
        "name": user.name,
        "email": user.email,
        "active": True,
    }
    users_db.append(new_user)
    return new_user

@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user: UserCreate):
    """Update an existing user"""
    for i, existing_user in enumerate(users_db):
        if existing_user["id"] == user_id:
            updated_user = {
                "id": user_id,
                "name": user.name,
                "email": user.email,
                "active": existing_user["active"],
            }
            users_db[i] = updated_user
            return updated_user
    raise HTTPException(status_code=404, detail="User not found")

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user"""
    for i, user in enumerate(users_db):
        if user["id"] == user_id:
            del users_db[i]
            return {"message": "User deleted successfully"}
    raise HTTPException(status_code=404, detail="User not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)