"""
Auth service — real operator accounts (bcrypt) + JWT sessions.

Replaces the old single shared password. Each operator is a `User`; JWTs survive
restarts (unlike the old in-memory token set). Clients are owned by a user, giving
per-operator tenant isolation.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from sqlalchemy.orm import Session

from db_models import User, Client
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_ALG = "HS256"


def _secret() -> str:
    # Prod should set JWT_SECRET. Dev falls back to a stable value derived from the
    # admin password so tokens remain valid across restarts.
    return settings.jwt_secret or f"dev-secret::{settings.admin_password}"


# ---- Password hashing (bcrypt) ----------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ---- JWT --------------------------------------------------------------------

def create_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(days=settings.jwt_expire_days),
    }
    return jwt.encode(payload, _secret(), algorithm=_ALG)


def decode_token(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALG])
        return int(payload["sub"])
    except Exception:
        return None


# ---- User store -------------------------------------------------------------

def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.strip().lower()).first()


def create_user(db: Session, email: str, password: str, name: str = "",
                is_superadmin: bool = False, role: str = "operator",
                client_slug: Optional[str] = None) -> User:
    user = User(
        email=email.strip().lower(),
        password_hash=hash_password(password),
        name=name or None,
        is_superadmin=is_superadmin,
        role="superadmin" if is_superadmin else role,
        client_slug=client_slug,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    # The very first user claims any legacy/unclaimed clients so existing data
    # (e.g. nexus, unihelp) belongs to someone and isolation can be demonstrated.
    if db.query(User).count() == 1:
        claimed = db.query(Client).filter(Client.owner_id.is_(None)).update(
            {Client.owner_id: user.id}, synchronize_session=False
        )
        db.commit()
        if claimed:
            logger.info(f"First user {user.email} claimed {claimed} legacy client(s)")
    return user


def authenticate(db: Session, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(db, email)
    if user and verify_password(password, user.password_hash):
        return user
    return None


def delete_user(db: Session, user_id: int) -> bool:
    user = db.get(User, user_id)
    if user is None:
        return False
    db.delete(user)
    db.commit()
    return True


def bootstrap_admin(db: Session) -> None:
    """Seed the first operator from env if no users exist (keeps the current
    operator logged in with the existing clients)."""
    if db.query(User).count() > 0:
        return
    email = settings.admin_email
    pw = settings.admin_password
    if not pw:
        logger.warning("No ADMIN_PASSWORD set; skipping admin bootstrap")
        return
    create_user(db, email=email, password=pw, name="Admin", is_superadmin=True)
    logger.info(f"Bootstrapped admin operator: {email}")
