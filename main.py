import json
import ccxt
from datetime import datetime, timedelta

ROW = 30
def create_dataset(path: str):
    exchange = ccxt.binance()
    start_date = datetime(2019, 1, 1)
    delta = (datetime.now() - start_date).days
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1d', since=int(start_date.timestamp() * 1000), limit=delta)
    first_eth_price = ohlcv[0][4]

    prices = [candle[4] for candle in ohlcv]

    prices = [(candle[4]/first_eth_price)-1.0 for candle in ohlcv]
    dataset = create_dataset_data(first_eth_price, prices, ROW)

    with open(path, "w") as file:
        file.write(json.dumps(dataset))

def create_dataset_data(start_price: float, prices, history: int):
    dataset = {
        "start": start_price
    }
    dataset["prices"] = []

    for i in range(history, len(prices)):
        previous = prices[i-history:i]
        next = prices[i]

        dataset["prices"].append({"history": previous, "next": next})
    
    return dataset

import torch
from torch import nn

MODEL = "model.pth"
DATASET = "dataset.json"

class Krad(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(ROW, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def build(self, x_train, y_train, optimizer, criterion, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(x_train)):
                inputs = torch.tensor(x_train[i], dtype=torch.float32)
                targets = torch.tensor([y_train[i]], dtype=torch.float32)
                
                optimizer.zero_grad()
                
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            print(f'[{epoch+1}/{epochs}] - {total_loss/len(x_train):.4f}')
    
    def predict(self, previous):
        inputs = torch.tensor(previous, dtype=torch.float32)
    
        output = self(inputs)
        
        return output.item()
    
def load_dataset(path: str):
    with open(path, "r") as file:
        dataset = json.loads(file.read())
    
    return (
        [prices["history"] for prices in dataset["prices"]],
        [prices["next"] for prices in dataset["prices"]],
        dataset["start"]
    )

def build(path: str, dataset: str) -> Krad:
    x_train, y_train, _ = load_dataset(dataset)

    model = Krad()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.build(x_train, y_train, optimizer, criterion, 200)

    torch.save(model.state_dict(), path)
    
    return model

def load(path: str) -> Krad:
    model = Krad()
    model.load_state_dict(torch.load(path))

    return model

model = load(MODEL)
x, y, start = load_dataset(DATASET)

last = x[-1]
last.pop(0)
last.append(y[-1])

prediction = model.predict(last)
print(f"predicted price - ${(prediction+1.0)*start}")