import pickle
import typing
from logging import getLogger

logger = getLogger()


def read_pickle(file_path: str) -> typing.Any:
    with open(str(file_path), "rb") as p:
        return pickle.load(p)


def write_pickle(data: typing.Any, file_path: str) -> None:
    with open(str(file_path), "wb") as p:
        pickle.dump(data, p, protocol=5)
