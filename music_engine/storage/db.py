from sqlalchemy import create_engine, String, Float, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from music_engine.config import DB_PATH


engine = create_engine(f"sqlite:///{DB_PATH}")


class Base(DeclarativeBase):
    pass


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String, unique=True)
    file_hash: Mapped[str] = mapped_column(String, unique=True)
    duration: Mapped[float] = mapped_column(Float)


def init_db():
    Base.metadata.create_all(engine)