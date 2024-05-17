import os
import pickle
from aslite.db import get_papers_db
from contextlib import contextmanager
import sqlite3, zlib, pickle, tempfile
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai
import google.ai.generativelanguage as glm

DATA_DIR = 'data'
VECTOR_DB_FILE = os.path.join(DATA_DIR, 'vector_db.p')


def embed_text(q_list, method="local"):
    embeddings = []
    if method == "gemini":
        for q in tqdm(q_list):
            model = 'models/embedding-001'
            title = "The next generation of AI for developers and Google Workspace"
            embedding = genai.embed_content(model=model,
                                            content=q,
                                            task_type="retrieval_document",
                                            title=title)
        embeddings.append(embedding)
    elif method == "openai":
        client = OpenAI()
        model="text-embedding-3-small"
        embeddings = client.embeddings.create(input = q_list, model=model).data
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Encodding")
        embeddings = model.encode(q_list, show_progress_bar=True)

    return embeddings

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


def get_papers_db_embedding(pdb, keys=("summary",), vectordb_file=VECTOR_DB_FILE):
    if os.path.exists(vectordb_file):
        vector_db = load_vectordb(vectordb_file)
    else:
        vector_db = {}
    
    list_pid = []
    list_to_embed = []
    for pid, p in pdb.items():
        if pid not in vector_db or keys not in vector_db[pid]:
            if pid not in vector_db:
                vector_db[pid] = {}

            to_embed = ""
            for k in keys:
                to_embed += "{}: {}\n".format(k, p[k])

            list_pid.append(pid)
            list_to_embed.append(to_embed)
    


    if len(list_to_embed):
        # Sentence encocoding
        embedded = embed_text(list_to_embed)
        embedded_dict = {}
        for pid, embd in zip(list_pid, embedded):
            embedded_dict[pid] = embd

        safe_pickle_dump(embedded_dict, vectordb_file)


if __name__ == "__main__":
    pdb = get_papers_db()
    get_papers_db_embedding(pdb)

