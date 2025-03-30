# AI_Agent_Investment_Advisory

## Overview
The AI Agent Investment Advisor is a sophisticated AI-powered financial advisory system designed to adapt to volatile market conditions, optimize portfolio recommendations, and ensure compliance with financial regulations. It dynamically processes real-time market data, retrains itself based on past errors, and provides explainable AI insights to users.

### Key Technologies
- **Transformer-based forecasting models** for stock price prediction.
- **Reinforcement learning** for portfolio optimization.
- **SHAP & LIME** for model explainability.
- **Regulatory compliance adherence** (SEBI, SEC guidelines).

## Features
- **Real-time Market Adaptation**: Processes live market data and historical trends.
- **Automated Error Handling & Retraining**: Enhances forecasting accuracy over time.
- **Explainability & Compliance**: Uses SHAP, LIME, and regulatory guidelines.
- **Portfolio Optimization**: Ensures risk-aware investment recommendations.
- **User Profile Management**: Customizes investment strategies based on user risk tolerance and preferences.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AI_Investment_Advisor.git
   cd AI_Investment_Advisor
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Upgrade pip and install dependencies:
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Project Structure
```
AI_Investment_Advisor/
│-- AI_Agent_IA.py             # Main AI advisory agent
│-- data_generation.py         # Generates mutual funds, stocks, and user profiles
│-- requirements.txt           # Project dependencies
│-- README.md                  # Project documentation
```

## Usage
1. **Generate Sample Data**:
   ```sh
   python data_generation.py
   ``` 
2. **Run the AI Investment Advisor**:
   ```sh
   python AI_Agent_IA.py --market live --risk high
   ```

## Requirements
The dependencies required for this project are listed in `requirements.txt`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contribution
Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## Contact
For any inquiries, reach out via [harshath142@gmail.com] or open an issue on GitHub.
