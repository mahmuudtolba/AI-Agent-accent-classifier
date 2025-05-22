import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from transformers import pipeline
from typing import List

import  os, time
import os
from dotenv import load_dotenv
from huggingface_hub import login
import time
import torch
import streamlit as st
from pydub import AudioSegment
import gdown
from huggingface_hub import snapshot_download
from pydantic import BaseModel


app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

######### Logging IN ##########

login("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")




class DataInput(BaseModel):
    video_url: List[str]


######### Download ML Models ##########
model_name = "ylacombe/accent-classifier"
local_model_dir = "./backend/pretrained_models/accent_modelylacombe" 

# Check if model already exists (e.g., config.json is there)

@app.get("/api/v1/check_model")
def check_model():
    # exists = os.path.exists("my_model_dir")
    # return {"model_exists": exists}

    if not os.path.exists(os.path.join(local_model_dir, "config.json")):

        download_the_model()
        return {"model_exists":"Model not found locally. Downloading..."}
        
    else:
        return {"model_exists": "Model already exists locally."}

def download_the_model():
    snapshot_download(
            repo_id=model_name,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False,
            force_download=False
        )

######## Download ENDS  #############



@app.get("/")
def read_root():
    return "Hello! I am up!!!"


@app.post("/api/v1/accent_classification")
async def accent_classification(data: DataInput):
    start = time.time()
    # Step 1: Download MP4

    video_url = str(data.video_url[0])
    video_path = "./backend/utils/video.mp4"
    audio_path = "./backend/utils/audio.wav"
    gdown.download(video_url, video_path ,quiet=False, fuzzy=True)

    # Step 2: Extract audio to WAV
    
    try:
        video = AudioSegment.from_file(video_path, format="mp4")
        # Export as WAV
        video.export(audio_path, format="wav")

   
    except Exception as e:
        return {"error": f"Failed to extract audio: {str(e)}"}
    
    
    # Step 3: Run the model
    try:
        pipe = pipeline("audio-classification", model=local_model_dir)
        scores = pipe(audio_path)
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # Cleanup
    os.remove(video_path)
    os.remove(audio_path)
    
    end = time.time()
    prediction_time = int((end-start)*1000)

    return scores
