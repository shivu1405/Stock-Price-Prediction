# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Stock prices change over time and are influenced by previous values. The goal of this experiment is to build a Recurrent Neural Network (RNN) that can learn patterns in historical stock prices and predict future prices.

The dataset used contains historical stock price data with features such as Open, High, Low, Close, and Volume. The Close price is typically used as the target variable for prediction.
<img width="1898" height="928" alt="image" src="https://github.com/user-attachments/assets/f7e6e11a-07f1-4a2b-974e-7ef6cee40b4f" />

<img width="1909" height="920" alt="image" src="https://github.com/user-attachments/assets/5c9a8d46-cfd9-450b-90a9-d43089e3cdeb" />

## Design Steps
### Step 1
Import required libraries such as NumPy, Pandas, Matplotlib, PyTorch, and Sklearn for data preprocessing, visualization, and model building.

### Step 2
Load the dataset, normalize the stock price values, and prepare sequential input data suitable for RNN training.

### Step 3
Define and train the Recurrent Neural Network model using PyTorch, then evaluate the model by comparing true stock prices with predicted stock prices.



## Program
#### Name: Shivasri.S
#### Register Number:212224220098
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        
        # Take last time step output
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model

num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    
    model.train()
    epoch_loss = 0
    
    for x_batch, y_batch in train_loader:
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        outputs = model(x_batch)
        
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')







```

## Output

### True Stock Price, Predicted Stock Price vs time

Include your plot here
<img width="912" height="513" alt="image" src="https://github.com/user-attachments/assets/1d65cf70-2a23-47b1-af51-b39447a743e4" />

### Predictions 

Include the predictions on test data
<img width="1134" height="639" alt="image" src="https://github.com/user-attachments/assets/2ac1f090-3643-45b1-b49e-596ffa8aa95a" />

## Result
Thus, a Recurrent Neural Network model for stock price prediction was successfully implemented using PyTorch, and the predicted values were compared with actual stock prices.

