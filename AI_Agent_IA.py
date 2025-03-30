import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from langchain_ollama import OllamaLLM  # Updated import
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load datasets
user_profiles = pd.read_csv("C:/Projects/Rudo_wealth/synthetic_user_profiles.csv")
stock_prices = pd.read_csv("C:/Projects/Rudo_wealth/synthetic_stock_prices.csv")
mutual_funds = pd.read_csv("C:/Projects/Rudo_wealth/synthetic_mutual_funds.csv")

# Ensure numerical data format
stock_prices = stock_prices.select_dtypes(include=[np.number]).fillna(method='ffill')
mutual_funds = mutual_funds.select_dtypes(include=[np.number]).fillna(method='ffill')

# Optimized NLP-based Query Understanding
llm = OllamaLLM(model="mistral", timeout=15)  # Reduced timeout for faster response
prompt = PromptTemplate.from_template("""
User Query: {query}
AI Response: Identify whether the query is related to stock forecasting, mutual fund forecasting, or portfolio optimization.
""")
nlp_chain = RunnableLambda(lambda query: llm.invoke(prompt.format(query=query)))

def analyze_query(query):
    try:
        return nlp_chain.invoke(query)  # Using invoke instead of run
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Optimized Stock Forecasting Model
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class GRUModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])  # Take last time step

# Train Model from Scratch
model = GRUModel().to(device)

def train_model(stock_data):
    global model
    free_cuda_memory()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    stock_data = np.array(stock_data, dtype=np.float32)
    seq_length = 10  # Define sequence length
    X, y = [], []
    
    for i in range(len(stock_data) - seq_length):
        X.append(stock_data[i:i+seq_length])
        y.append(stock_data[i+seq_length])
    
    X = torch.tensor(np.array(X), dtype=torch.float32).view(-1, seq_length, stock_data.shape[1]).to(device, non_blocking=True)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, stock_data.shape[1]).to(device, non_blocking=True)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(50):  # Reduced epochs to prevent memory overflow
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    torch.save(model.state_dict(), "gru_model.pth")
    model.eval()

# Train Model
train_model(stock_prices.values)

def forecast_stock_prices(data):
    model.eval()
    data = np.array(data, dtype=np.float32).reshape(1, 10, data.shape[1])  # Ensure correct shape
    data = torch.tensor(data, dtype=torch.float32).to(device, non_blocking=True)
    
    with torch.no_grad():
        return model(data).cpu().numpy().flatten().tolist()

# Optimized Portfolio Allocation (Use only last 6 months of data)
def optimize_portfolio(prices_df):
    recent_prices = prices_df.tail(180)  # Use last 180 days
    mu = mean_historical_return(recent_prices)
    S = CovarianceShrinkage(recent_prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    return ef.clean_weights()

# AI Investment Advisor
def investment_advisor(query, stock_data, mutual_fund_data, prices_df):
    nlp_response = analyze_query(query)
    
    if "stock" in query.lower():
        forecasted_price = forecast_stock_prices(stock_data)
        return {"Query Response": nlp_response, "Stock Forecast": forecasted_price}
    elif "mutual fund" in query.lower():
        forecasted_price = forecast_stock_prices(mutual_fund_data)
        return {"Query Response": nlp_response, "Mutual Fund Forecast": forecasted_price}
    elif "portfolio" in query.lower() or "investment allocation" in query.lower():
        optimal_allocation = optimize_portfolio(prices_df)
        return {"Query Response": nlp_response, "Optimal Portfolio Allocation": optimal_allocation}
    else:
        return {"Query Response": "Please specify stock forecasting, mutual fund forecasting, or portfolio optimization."}

# Continuous Conversation with the AI Bot
if __name__ == "__main__":
    print("Welcome to the AI Investment Advisor!")
    print("You can have a continuous conversation. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your investment query: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the AI Investment Advisor. Goodbye!")
            break
        response = investment_advisor(user_query, stock_prices.values[-10:], mutual_funds.values[-10:], stock_prices)
        print("\nAI Response:")
        for key, value in response.items():
            print(f"{key}: {value}")
        print("\n----------------------------------------\n")
