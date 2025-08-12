# **Agentic Inventory Optimizer**  
Multi-Agent Reinforcement Learning for Multi-Warehouse Inventory Management

## 📌 Overview
This project implements an **Agentic Workflow System** for optimizing multi-warehouse inventory operations using **Reinforcement Learning (RL)**.  
It integrates:
- **Value-Based Learning** → Deep Q-Network (DQN)  
- **Policy Gradient Methods** → Proximal Policy Optimization (PPO)  
- **Exploration Strategies** → Upper Confidence Bound (UCB) policy selection  
- **Custom Agentic Tools** → Cost simulation, decision explanation, and warehouse Q&A  

The goal is to **minimize total inventory costs** (holding + stockout + ordering) while **maintaining high service levels** under uncertain demand.

---

## 🚀 Features
- **Two RL approaches**: DQN & PPO
- **Multi-agent orchestration** with policy selection
- **Fallback mechanism** for safety
- **Custom tools** for:
  - Cost simulation
  - Decision explanation
  - Warehouse Q&A
- **Streamlit dashboard** for visualization
- **Baseline comparisons** with heuristic policies
- **Exportable reports** with learning curves & performance breakdowns

---

## 📂 Repository Structure
```
agentic-inventory-optimizer/
├── agents/                  # RL agents, forecaster, orchestrator, policy selector
├── custom_tools/             # Cost simulation, dashboard export, decision explainer, QA
├── env/                      # Inventory environment & wrappers
├── demo/                     # Streamlit UI
├── rl/                       # Training & evaluation scripts
├── results/                  # Models, evaluation JSONs, and visualizations
├── tests/                    # Unit tests for agents & tools
└── README.md                 # This file
```

---

## ⚙️ Installation
1. **Clone the repo**
```bash
git clone https://github.com/anumohan10/agentic-inventory-optimizer.git
cd agentic-inventory-optimizer
```
2. **Create virtual environment & install dependencies**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## ▶️ Usage

### **1. Train an RL Agent**
#### DQN
```bash
python -m rl.train_rl_agent --algo dqn --episodes 10000     --target_service 0.92 --below_target_mult 12.0 --seed 42
```
#### PPO
```bash
python -m rl.train_rl_agent --algo ppo --episodes 8000     --target_service 0.92 --below_target_mult 8.0 --seed 0
```

### **2. Evaluate a Model**
```bash
python -m rl.evaluate_agent --algo ppo --episodes 100     --model_path results/models/ppo_best_98_service.zip
```

### **3. Run the Dashboard**
```bash
streamlit run demo/app.py
```

---

## 📊 Results

| Policy     | Total Cost | Service Level | Notes       |
|------------|------------|--------------|-------------|
| **PPO**   | $6,504     | 98.85%       | 🥇 Best     |
| **DQN**   | $6,897     | 98.33%       | 🥈 Excellent|
| Heuristic | ~$7,500–8,500 | 85–92%    | 📊 Baseline |

---

## 🧠 Agentic System Workflow
```mermaid
flowchart TD
    subgraph UI[UI / CLI]
    end
    UI --> O[Orchestrator]
    O --> PS[Policy Selector]
    PS -->|Choose| RL[RL Agent (PPO/DQN)]
    PS -->|Fallback| HB[Heuristic Baseline]
    O --> F[Forecaster]
    RL --> ENV[Inventory Environment]
    HB --> ENV
    ENV --> COST[Cost Calculation]
    COST --> REW[Reward Function]
```

---

## 📌 Key Achievements
- **DQN** improved service from 69.7% → 98.33%  
- **Cost reduction** of 39% for DQN after tuning  
- **PPO** achieved optimal cost-service trade-off  
- **Agentic orchestration** with policy switching and fallback safety

---

## 📈 Future Improvements
- Multi-agent RL (one per warehouse)
- Continuous action spaces
- Integration with real demand forecasting models
- Transfer learning between warehouses
- Testing with real-world supply chain datasets

---

## 📜 License
MIT License © 2025 [Anusree Mohanan]
