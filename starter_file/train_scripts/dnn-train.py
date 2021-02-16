import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import copy
import argparse
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from azureml.core.run import Run

run = Run.get_context()

# separar data / target en columnas diferentes
# construir conjuntos train / test
# preprocesar variables numericas / categoricas
# construir un pipeline
# construir una red neuronal (tener en cuenta el desbalanceo de clases)
# posibles parametros para buscar: lr, num_epochs, layer1, layer2, decay_rate

# Imbalanced class: No: 0.838776 / Yes: 0.161224
# No missing values
# Column Over18 constant: "Y"
# Column EmployeeCount constant: 1
# EmployeeNumber row identifier

class DnnClassifier(nn.Module):
    def __init__(self, num_layer1=30, num_layer2=15):
        super(DnnClassifier, self).__init__()
    
        self.layer_1 = nn.Linear(52, num_layer1)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.1)
        self.layer_2 = nn.Linear(num_layer1, num_layer2)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.1)
        self.layer_output = nn.Linear(num_layer2, 2)
        
    def forward(self, inputs):
        out = self.relu_1(self.layer_1(inputs))
        out = self.dropout_1(out)
        out = self.relu_2(self.layer_2(out))
        out = self.dropout_2(out)
        out = self.layer_output(out)
        
        return out
    
def load_data(data_path="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    df = pd.read_csv(data_path, index_col="EmployeeNumber")

    return df

def preprocess_data(df):
    df.drop(["Over18","EmployeeCount"], axis=1, inplace=True)
    
    target = df["Attrition"]
    df.drop("Attrition", axis=1, inplace=True)
    target = LabelEncoder().fit_transform(target)
    
    X_train, X_test, y_train, y_test = train_test_split(df, 
                                                        target, 
                                                        test_size=0.2, 
                                                        stratify=target, 
                                                        random_state=42, 
                                                        shuffle=True)
    
    num_columns = df.select_dtypes("number").columns
    cat_columns = df.columns.difference(num_columns)
    
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(sparse=False))])
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    
    dnn_pipeline = ColumnTransformer([('numeric', num_pipeline, num_columns), 
                                      ('categoric', cat_pipeline, cat_columns)])
    
    X_train = dnn_pipeline.fit_transform(X_train)
    X_test = dnn_pipeline.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def fine_tune_model(num_epochs, learning_rate, num_layer1, num_layer2):
    model = DnnClassifier(num_layer1, num_layer2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    
    run.log("num_epochs", np.int(num_epochs))
    run.log("learning_rate", np.float(learning_rate))
    run.log("num_layer1", np.int(num_layer1))
    run.log("num_layer2", np.int(num_layer2))
    
    model = train_model(model, optimizer, criterion, num_epochs)
    
    return model

def train_model(model, optimizer, criterion, num_epochs):
    
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    X_train, X_test = torch.from_numpy(X_train.astype(np.float32)), \
                      torch.from_numpy(X_test.astype(np.float32))
    y_train, y_test = torch.from_numpy(y_train).long(), \
                      torch.from_numpy(y_test).long()
    
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    
    running_train_loss = []
    running_test_loss = []
    
    running_train_metric = []
    running_test_metric = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0
    
    for iter in range(num_epochs):
        optimizer.zero_grad()
        
        # train phase
        model.train()
        output = model(X_train)
        
        train_loss = criterion(output, y_train)
        _, preds = torch.max(output, 1)
        train_acc = torch.mean((preds == y_train).float())
        
        running_train_metric.append(train_acc)
        running_train_loss.append(train_loss.item())
        
        # scheduler.step()
        train_loss.backward()
        optimizer.step()
        
        # test phase
        model.eval()
        with torch.no_grad():
            output = model(X_test)
            test_loss = criterion(output, y_test)
            
            _, preds = torch.max(output, 1)
            # test_acc = torch.mean((preds == y_test).float())
            auc_weighted = roc_auc_score(y_test.cpu().numpy(), 
                                         preds.cpu().numpy(), 
                                         average="weighted")
            
            running_test_metric.append(auc_weighted)
            running_test_loss.append(test_loss.item())
            
            if auc_weighted > best_metric:
                best_metric = auc_weighted
                best_model_wts = copy.deepcopy(model.state_dict())
            
        
        # print("Train loss: {}, Test loss: {}".format(running_train_loss[-1], 
        #                                              running_test_loss[-1],))
        # print("Epoch: {}, Train metric: {}, Test metric: {}, Best metric: {}".format(iter + 1,
        #                                                                               running_train_metric[-1],
        #                                                                               running_test_metric[-1],
        #                                                                               best_metric))
        run.log('best_test_acc', np.float(best_metric))
        
    model.load_state_dict(best_model_wts)
    return model
    
def main():
    
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--layer1", type=int, default=30, help="layer 1 neurons")
    parser.add_argument("--layer2", type=int, default=15, help="layer 2 neurons")
    
    args = parser.parse_args()
    
    
    model = fine_tune_model(args.num_epochs, args.learning_rate, args.layer1, 
                            args.layer2)
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, 'model.pt'))
    
if __name__ == "__main__":
    main()