import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

api_keys = [
    "543c7086-c880-45de-8bce-6c9c906293bb"
]  

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )

app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

from embed import embed_audio
import tempfile
import os

@app.post("/embed/", dependencies=[Depends(api_key_auth)])
async def embed(file: UploadFile):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.file.read())
        try:
            embedding = embed_audio(temp_file_path)
            return json.dumps({"embedding": embedding.tolist()})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")