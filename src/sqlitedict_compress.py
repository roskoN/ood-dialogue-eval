import pickle
import sqlite3
import zlib


def my_encode(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))


def my_decode(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))
