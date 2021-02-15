import os
import torch
import torch.nn as nn 
import json

from azureml.core.model import Model

def init():
    global model
    model_path = os.path.join(os.getenv("AZURE_MODEL_DIR"), 'model.pt')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    
def run(input_data):
    input_data = torch.tensor(json.loads(input_data)['data'])
    
    #get predictions
    with torch.no_grad():
        output = model(input_data)
        classes = ['No', 'Yes']
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)
        
    result = {"label": classes[index], 
              "probability": str(pred_probs[index])}
    return result