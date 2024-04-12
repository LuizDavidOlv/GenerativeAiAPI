import asyncio
from fastapi import Depends, HTTPException
from fastapi import APIRouter
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt import PyJWTError

router = APIRouter(
    prefix="/authorization",
    tags=["Authorization"]
)

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

users = {
    "username": {
        "username": "username",
        "password": "password"
    }
}

@router.post("/login")
async def login(username: str, password: str) -> dict:
    user = users.get(username)
    if user is None or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = generate_token(username)
    return {"access_token": token, "token_type": "bearer"}

    
def generate_token(username: str) -> str:
    payload = {"sub": username}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="login"))) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": f"Hello, {current_user}! This is a protected endpoint."}
  

async def main():
    jwt = await login("username", "password")
    result = await protected_route(jwt["access_token"])
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

