import os
import pickle
from dotenv import load_dotenv
from aslite.db import get_papers_db
from aslite.embedding import embed_text
from contextlib import contextmanager
import sqlite3, zlib, pickle, tempfile
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai
import google.ai.generativelanguage as glm
from pinecone import Pinecone, ServerlessSpec


load_dotenv()  # Loads from '.env' file 


def load_vectordb(vectordb_file):
    """ loads the features dict from disk """
    with open(vectordb_file, 'rb') as f:
        vectordb = pickle.load(f)
    return vectordb

@contextmanager
def _tempfile(*args, **kws):
    """ Context for temporary file.
    Will find a free temporary filename upon entering
    and will try to delete the file on leaving
    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e

@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """ Open temporary file object that atomically moves to destination upon
    exiting.
    Allows reading and writing to and from the same filename.
    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop('fsync', False)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)

def safe_pickle_dump(obj, fname):
    """
    prevents a case where one process could be writing a pickle file
    while another process is reading it, causing a crash. the solution
    is to write the pickle file to a temporary file and then move it.
    """
    with open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1) # -1 specifies highest binary protocol


def get_papers_db_embedding(pdb, keys=("summary",)):
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    index = pc.Index("litrev")
    namespace = " ".join(keys)
    idx_set = []


    pid_list, pid_text = [], []

    for idx in index.list(namespace=namespace):
        idx_set += idx

    idx_set = set(idx_set)

    for pid, p in pdb.items():
        if pid not in idx_set:
            to_embed = ""
            for k in keys:
                to_embed += "{}: {}\n".format(k, p[k])

            pid_list.append(pid)
            #print(pid)
            pid_text.append(to_embed)

    if pid_text:
        embed_list = embed_text(pid_text)
        for pid, embd in zip(pid_list, embed_list):

            index.upsert(vectors=[{
                "id": pid,
                "values": embd
            }], namespace=namespace)

if __name__ == "__main__":
    pdb = get_papers_db()
    get_papers_db_embedding(pdb)

