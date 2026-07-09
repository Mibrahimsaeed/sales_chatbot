# from fastapi import APIRouter
# from pydantic import BaseModel

# router = APIRouter(
#     tags=["Auth"]
# )


# class LoginRequest(BaseModel):
#     name: str


# @router.post("/token")
# async def create_token(data: LoginRequest):
#     return {
#         "access_token": "test-token",
#         "token_type": "bearer",
#         "user": data.name
#     }



from fastapi import APIRouter
from pydantic import BaseModel
from app.core.security import create_token

router = APIRouter(
    tags=["Auth"]
)


class LoginRequest(BaseModel):
    name: str


@router.post("/token")
async def create_access_token(data: LoginRequest):
    token = create_token(
        {
            "sub": data.name
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": data.name
    }


