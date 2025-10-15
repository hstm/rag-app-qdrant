# RAG System with Anthropic API (Claude), FastAPI Web Interface, Semantic Chunking, and PDF Upload

## Features

- ✅ Semantic chunking
- ✅ PDF upload support
- ✅ Prompt caching (90% cost savings!)
- ✅ Smart relevance filtering
- ✅ Simple Web UI
- ✅ [Qdrant](https://qdrant.tech/) as vector database
- ✅ Type-safe [FastAPI](https://fastapi.tiangolo.com/) routes
- ✅ Marked.js for Markdown rendering

## Install

``` python -m venv rag-app ```

``` source rag-app/bin/activate ```

### App version using sentence-transformers:

```pip install fastapi uvicorn python-multipart anthropic sentence-transformers qdrant-client pypdf```

### App version using transformers directly:

```pip install fastapi uvicorn python-multipart anthropic transformers torch qdrant-client pypdf```

### App version using transformers with custom progress bar:

```pip install fastapi uvicorn python-multipart anthropic transformers torch qdrant-client pypdf tqdm```

## Run

### Export the Anthropic API Key:

``` export ANTHROPIC_API_KEY='your-api-key' ```

### Run app version using sentence-transformers:

``` uvicorn app_qdrant_fastapi:app --reload ```

### Run app version using transformers directly:

``` uvicorn app_qdrant_fastapi_tf:app --reload ```

### Run app version using transformers with custom progress bar:

``` uvicorn app_qdrant_fastapi_tf_prog:app --reload ```

depending on the version you want to start.

>[!NOTE]
>When you enable the progress bar, you will notice a warning like this on shutdown:
>
>```
>/home/stahlhe2/.local/share/pypoetry/python/cpython@3.12.9/lib/python3.12/multiprocessing/>resource_tracker.py:255: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects >to clean up at shutdown
>  warnings.warn('resource_tracker: There appear to be %d '
>```
>
>The semaphore leak warning is a known issue with sentence-transformers and transformers libraries >when using multiprocessing, you can safely ignore it. The resources are still freed by the OS when the process exits.

### Open the browser:

``` Open http://localhost:8000 in your browser```


## Documentation

You will find the Swagger/OpenAPI docs under

http://localhost:8000/docs

## Optimizations

You can optimize the startup time by skipping the example documents. Just comment out this line:

``` rag.add_documents(example_docs) ```
