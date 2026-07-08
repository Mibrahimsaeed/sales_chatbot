import datetime
from jose import jwt, JWTError
from app.core.config import settings


def create_token(payload: dict, expires_minutes: int = 60) -> str:
    """Dev/testing only — mints a token with the same secret get_current_user
    validates. Once the real dashboard issues tokens, this becomes unused;
    keep it around for local testing regardless."""
    to_encode = payload.copy()
    to_encode["exp"] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes)
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    """Point jwt_secret/jwt_algorithm at whatever your dashboard's existing
    auth already uses — this API validates those tokens, it doesn't issue
    its own."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None