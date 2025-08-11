# demo/app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.inventory_env import InventoryEnv
from agents.demand_forecast_agent import DemandForecastAgent
from agents.restock_recommender_agent import RestockRecommenderAgent
from agents.rl_agent import RLAgent
from agents.policy_selector import UCBBandit
from agents.orchestrator import Orchestrator, SafetyFallback
from custom_tools.dashboard_export_tool import DashboardExportTool

# Page config
st.set_page_config(
    page_title="🏭 Warehouse Inventory Optimizer",
    page_icon="📦",
    layout="wide"
)

# Initialize session state
if "simulation_data" not in st.session_state:
    st.session_state.simulation_data = None
if "rollouts_df" not in st.session_state:
    st.session_state.rollouts_df = None
if "summary_json" not in st.session_state:
    st.session_state.summary_json = {}

# Helper functions
def load_summary(path):
    """Load evaluation summary JSON."""
    with open(path) as f:
        return json.load(f)

def load_jsonl(path):
    """Load JSONL rollout data."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def run_simulation(policy_type, episodes=10, config=None):
    """Run simulation with selected policy."""
    env = InventoryEnv(
        n_warehouses=config.get("n_warehouses", 2),
        holding_cost=config.get("holding_cost", 1.0),
        stockout_cost=config.get("stockout_cost", 10.0),
        order_cost=config.get("order_cost", 2.0)
    )
    
    # Initialize agents
    forecaster = DemandForecastAgent()
    
    # Load policies
    policies = {}
    
    if policy_type == "Orchestrated (UCB)":
        # Load multiple policies for orchestration
        if Path("results/models/ppo_W2.zip").exists():
            rl_agent_balanced = RLAgent()
            rl_agent_balanced.load("results/models/ppo_W2.zip")
            policies["ppo_balanced"] = lambda s: rl_agent_balanced.predict(s)
            
        if Path("results/models/ppo_high_service/ppo_W2.zip").exists():
            rl_agent_high = RLAgent()
            rl_agent_high.load("results/models/ppo_high_service/ppo_W2.zip")
            policies["ppo_high"] = lambda s: rl_agent_high.predict(s)
            
        # Add heuristic baseline
        heuristic = RestockRecommenderAgent(policy="order_up_to_s")
        policies["heuristic"] = lambda s: heuristic.recommend(s, warehouse_id=0)
        
        # Setup bandit and orchestrator
        bandit = UCBBandit(list(policies.keys()))
        fallback = SafetyFallback(target_service=0.92)
        orchestrator = Orchestrator(policies, bandit, forecaster, fallback)
        
    results = []
    for ep in range(episodes):
        state = env.reset()
        episode_cost = 0
        episode_data = []
        
        for t in range(50):  # 50 steps per episode
            if policy_type == "Orchestrated (UCB)":
                actions = []
                meta_list = []
                for w in range(env.n_warehouses):
                    action, meta = orchestrator.step(state, w)
                    actions.append(action)
                    meta_list.append(meta)
                actions = np.array(actions)
            else:
                # Single policy execution
                if policy_type == "PPO-Balanced":
                    agent = RLAgent()
                    agent.load("results/models/ppo_W2.zip")
                    actions = agent.predict(state)
                elif policy_type == "DQN":
                    agent = RLAgent()
                    agent.load("results/models/dqn_W2.zip")
                    actions = agent.predict(state)
                else:  # Heuristic
                    agent = RestockRecommenderAgent(policy="order_up_to_s")
                    actions = np.array([agent.recommend(state, w) for w in range(env.n_warehouses)])
                    
            state, reward, done, info = env.step(actions)
            episode_cost += -reward
            
            episode_data.append({
                "episode": ep,
                "step": t,
                "total_cost": -reward,
                "holding_cost": info.get("holding_cost", 0),
                "stockout_cost": info.get("stockout_cost", 0),
                "order_cost": info.get("order_cost", 0),
                "service_level": info.get("service_level", 0),
                "policy": meta_list[0]["policy"] if policy_type == "Orchestrated (UCB)" else policy_type
            })
            
        if policy_type == "Orchestrated (UCB)":
            orchestrator.end_episode(episode_cost, info.get("service_level", 0))
            
        results.extend(episode_data)
        
    return pd.DataFrame(results)

def explain_decision(state, action, policy_type):
    """Generate explanation for a decision."""
    stock = state[:2]  # First 2 values are warehouse stocks
    recent_demand = state[2] if len(state) > 2 else 50
    
    explanation = f"📊 **Decision Context:**\n"
    explanation += f"- Current Stock: W1={stock[0]:.0f}, W2={stock[1]:.0f}\n"
    explanation += f"- Recent Demand: {recent_demand:.0f}\n"
    explanation += f"- Action Taken: Order {action:.0f} units\n\n"
    
    explanation += f"💡 **Reasoning:**\n"
    if action > 60:
        explanation += "Large order placed due to low stock relative to demand. "
        explanation += "Preventing stockouts is prioritized given high penalty costs."
    elif action > 30:
        explanation += "Moderate order to maintain safety stock levels. "
        explanation += "Balancing holding costs with service level requirements."
    elif action > 0:
        explanation += "Small replenishment order to top up inventory. "
        explanation += "Current stock levels are adequate for expected demand."
    else:
        explanation += "No order placed as current stock is sufficient. "
        explanation += "Minimizing holding costs while maintaining service levels."
        
    return explanation

# Streamlit UI
st.title("🏭 Agentic Inventory Optimizer")
st.markdown("Multi-warehouse inventory management with Reinforcement Learning")

# Create tabs
tab_sim, tab_analytics, tab_chat = st.tabs(["🎮 Simulate", "📊 Analytics", "💬 Ask the Warehouse"])

# Tab 1: Simulation
with tab_sim:
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("⚙️ Configuration")
        n_warehouses = st.slider("Number of Warehouses", 1, 5, 2)
        holding_cost = st.slider("Holding Cost ($/unit/day)", 0.5, 5.0, 1.0)
        stockout_cost = st.slider("Stockout Cost ($/unit)", 5.0, 50.0, 10.0)
        order_cost = st.slider("Order Cost ($/unit)", 1.0, 10.0, 2.0)
        
    with col2:
        st.subheader("🤖 Policy Selection")
        policy_type = st.selectbox(
            "Select Policy",
            ["Heuristic (Order-up-to-S)", "DQN", "PPO-Balanced", "Orchestrated (UCB)"]
        )
        episodes = st.slider("Simulation Episodes", 1, 50, 10)
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                config = {
                    "n_warehouses": n_warehouses,
                    "holding_cost": holding_cost,
                    "stockout_cost": stockout_cost,
                    "order_cost": order_cost
                }
                df = run_simulation(policy_type, episodes, config)
                st.session_state.simulation_data = df
                st.session_state.rollouts_df = df
                st.success(f"✅ Simulation complete! Ran {episodes} episodes.")
                
    with col3:
        st.subheader("📈 Quick Stats")
        if st.session_state.simulation_data is not None:
            df = st.session_state.simulation_data
            metrics = df.groupby("episode").agg({
                "total_cost": "sum",
                "service_level": "mean"
            }).mean()
            
            st.metric("Avg Total Cost", f"${metrics['total_cost']:.2f}")
            st.metric("Avg Service Level", f"{metrics['service_level']:.1%}")
            
            # Show episode breakdown
            episode_costs = df.groupby("episode")["total_cost"].sum()
            st.line_chart(episode_costs, use_container_width=True)
            
    # Decision Explainer Section
    if st.session_state.simulation_data is not None:
        st.divider()
        st.subheader("🔍 Decision Explainer")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_episode = st.selectbox("Select Episode", range(episodes))
            selected_step = st.selectbox("Select Step", range(50))
            
        with col2:
            # Get data for selected step
            step_data = df[(df["episode"] == selected_episode) & (df["step"] == selected_step)]
            if not step_data.empty:
                # Mock state/action for explanation (you'd get real ones from env)
                mock_state = np.array([45, 38, 52, 48, 50])  
                mock_action = 40
                explanation = explain_decision(mock_state, mock_action, policy_type)
                st.markdown(explanation)

# Tab 2: Analytics
with tab_analytics:
    st.subheader("📊 Performance Analytics")
    
    # Load evaluation results
    eval_files = list(Path("results/eval").glob("*.json"))
    
    if eval_files:
        # Load all evaluation summaries
        summaries = {}
        for file in eval_files:
            name = file.stem
            summaries[name] = load_summary(file)
            
        # Create comparison dataframe
        comparison_data = []
        for name, data in summaries.items():
            comparison_data.append({
                "Policy": name,
                "Total Cost": data.get("mean_total_cost", 0),
                "Service Level": data.get("mean_service", 0),
                "Holding Cost": data.get("mean_holding", 0),
                "Stockout Cost": data.get("mean_stockout", 0),
                "Order Cost": data.get("mean_order", 0)
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Display metrics
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Cost breakdown bar chart
            fig_costs = go.Figure()
            for policy in comp_df["Policy"]:
                row = comp_df[comp_df["Policy"] == policy].iloc[0]
                fig_costs.add_trace(go.Bar(
                    name=policy,
                    x=["Holding", "Stockout", "Order"],
                    y=[row["Holding Cost"], row["Stockout Cost"], row["Order Cost"]]
                ))
            fig_costs.update_layout(title="Cost Breakdown by Policy", barmode="group")
            st.plotly_chart(fig_costs, use_container_width=True)
            
        with col2:
            # Service vs Cost scatter
            fig_scatter = px.scatter(
                comp_df, 
                x="Total Cost", 
                y="Service Level",
                text="Policy",
                title="Service Level vs Total Cost Trade-off",
                labels={"Service Level": "Service Level (%)"}
            )
            fig_scatter.update_traces(textposition="top center", marker_size=12)
            fig_scatter.update_yaxis(tickformat=".0%")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        # Detailed comparison table
        st.subheader("📋 Detailed Comparison")
        st.dataframe(
            comp_df.style.format({
                "Total Cost": "${:.2f}",
                "Service Level": "{:.1%}",
                "Holding Cost": "${:.2f}",
                "Stockout Cost": "${:.2f}",
                "Order Cost": "${:.2f}"
            }).highlight_min(subset=["Total Cost"], color="lightgreen")
            .highlight_max(subset=["Service Level"], color="lightblue"),
            use_container_width=True
        )
        
        # Learning curves if available
        if Path("results/eval/ppo_rollouts.jsonl").exists():
            st.subheader("📈 Learning Progress")
            rollouts_df = load_jsonl("results/eval/ppo_rollouts.jsonl")
            
            # Rolling average of episode costs
            episode_costs = rollouts_df.groupby("episode")["total_cost"].sum().rolling(10).mean()
            
            fig_learning = go.Figure()
            fig_learning.add_trace(go.Scatter(
                x=episode_costs.index,
                y=episode_costs.values,
                mode="lines",
                name="Rolling Avg Cost (window=10)"
            ))
            fig_learning.update_layout(
                title="Learning Curve: Episode Cost Over Time",
                xaxis_title="Episode",
                yaxis_title="Total Cost ($)"
            )
            st.plotly_chart(fig_learning, use_container_width=True)
            
    else:
        st.warning("No evaluation results found. Please run evaluation first.")

# Tab 3: Chat/Q&A
with tab_chat:
    st.subheader("💬 Ask the Warehouse")
    st.markdown("Ask questions about the simulation results, costs, and decisions.")
    
    # Simple Q&A system
    question = st.text_input("Ask a question:", placeholder="e.g., Which episode had the worst stockouts?")
    
    if question and st.session_state.rollouts_df is not None:
        df = st.session_state.rollouts_df
        q_lower = question.lower()
        
        # Pattern matching for common questions
        if "worst stockout" in q_lower or "highest stockout" in q_lower:
            episode_stockouts = df.groupby("episode")["stockout_cost"].sum()
            worst_ep = episode_stockouts.idxmax()
            worst_val = episode_stockouts.max()
            st.success(f"📊 Episode {worst_ep} had the highest stockout cost: ${worst_val:.2f}")
            
        elif "best episode" in q_lower or "lowest cost" in q_lower:
            episode_costs = df.groupby("episode")["total_cost"].sum()
            best_ep = episode_costs.idxmin()
            best_val = episode_costs.min()
            st.success(f"🏆 Episode {best_ep} had the lowest total cost: ${best_val:.2f}")
            
        elif "average" in q_lower and "service" in q_lower:
            avg_service = df["service_level"].mean()
            st.info(f"📈 Average service level across all episodes: {avg_service:.1%}")
            
        elif "average" in q_lower and "cost" in q_lower:
            avg_cost = df.groupby("episode")["total_cost"].sum().mean()
            st.info(f"💰 Average total cost per episode: ${avg_cost:.2f}")
            
        elif "holding" in q_lower and "high" in q_lower:
            st.info("📦 Holding costs are high because the policy maintains safety stock to avoid "
                   "expensive stockouts. Consider adjusting the safety stock multiplier or "
                   "implementing dynamic thresholds based on demand volatility.")
            
        elif "why" in q_lower and "order" in q_lower:
            st.info("🤔 Orders are triggered when: 1) Stock falls below safety threshold, "
                   "2) Predicted demand exceeds current inventory, or 3) The RL policy "
                   "anticipates future demand spikes based on historical patterns.")
            
        else:
            st.write("**Try asking:**")
            st.write("- Which episode had the worst stockouts?")
            st.write("- What's the average service level?")
            st.write("- Why are holding costs high?")
            st.write("- Which episode had the lowest cost?")
            
    elif question:
        st.warning("Please run a simulation first to enable Q&A.")
        
    # Display recent metrics if available
    if st.session_state.rollouts_df is not None:
        st.divider()
        st.subheader("📊 Current Session Statistics")
        
        df = st.session_state.rollouts_df
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Episodes Run", df["episode"].nunique())
            st.metric("Avg Service Level", f"{df['service_level'].mean():.1%}")
            
        with col2:
            total_costs = df.groupby("episode")["total_cost"].sum()
            st.metric("Best Episode Cost", f"${total_costs.min():.2f}")
            st.metric("Worst Episode Cost", f"${total_costs.max():.2f}")
            
        with col3:
            st.metric("Avg Holding Cost", f"${df['holding_cost'].mean():.2f}")
            st.metric("Avg Stockout Cost", f"${df['stockout_cost'].mean():.2f}")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    🤖 Agentic Inventory Optimizer | Reinforcement Learning Final Project<br>
    PPO vs DQN vs Heuristic Policies with UCB Orchestration
    </div>
""", unsafe_allow_html=True)