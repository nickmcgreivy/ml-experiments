import pickle
import io
from functools import lru_cache

from fastapi import FastAPI, UploadFile
from torchvision import models, transforms # type: ignore
import torch
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    epochs = checkpoint['epochs']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    train_records = checkpoint['train_records']
    val_records = checkpoint['val_records']
    return (epochs, model_state_dict, optimizer_state_dict, 
            train_records, val_records)

@lru_cache
def load_imagenet_mapping():
    with open('idx_to_label.pkl', 'rb') as f:
        idx_to_label = pickle.load(f)
    return idx_to_label

@lru_cache
def load_model():
    _, trained_model_state_dict, _, _, _ =\
        load_checkpoint('imagenet-checkpoint.pt')
    model = models.resnet50()
    model.load_state_dict(trained_model_state_dict)
    model.eval()
    return model

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

@app.get("/")
def serve_frontend():
    return FileResponse("frontend.html")

@app.post('/predict/')
async def predict(k: int, imgfile: UploadFile):
    idx_to_label = load_imagenet_mapping()
    model = load_model()

    contents = await imgfile.read()
    await imgfile.close()

    try:
        image = Image.open(io.BytesIO(contents))
    except IOError:
        return {"message": "Couldn't open image file"}

    X = transform(image).to(device).unsqueeze(0)

    with torch.inference_mode():
        logits = model(X).squeeze(0)
    
    _, top_k_idx = torch.topk(logits, k, dim=-1)
    predicted_names = [idx_to_label[k.item()] for k in top_k_idx]
    return {'predicted': predicted_names}

@app.get('/check_device/')
def check_device():
    return {'device': str(device)}