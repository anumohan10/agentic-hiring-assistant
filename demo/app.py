# demo/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from env.inventory_env import InventoryEnv
from agents.restock_recommender_agent import fixed_order_policy, order_up_to_S, ForecastAwarePolicy
from agents.demand_forecast_agent import MovingAverageForecaster, EMAForecaster

# Check for dependencies at startup
try:
    import stable_baselines3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Try to import RL agent if it exists
try:
    from agents.rl_agent import RLInventoryAgent
except ImportError:
    RLInventoryAgent = None

from custom_tools.cost_simulation_tool import CostParams
from custom_tools.dashboard_export_tool import DashboardExporter

# Try to import DecisionExplainer
try:
    from custom_tools.decision_explainer_tool import DecisionExplainerTool
    DecisionExplainer = DecisionExplainerTool
except ImportError:
    try:
        from custom_tools.decision_explainer_tool import DecisionExplainer  # pragma: no cover
    except ImportError:
        # Simple fallback explainer
        class DecisionExplainer:
            def explain(self, obs, action_idx, env):
                stock = obs[0] if len(obs) > 0 else 0
                # Fallback grid (won't affect sim; just for explanation text)
                _levels = [0, 10, 20, 30, 40, 50, 60, 70, 80]
                order_qty = _levels[action_idx] if 0 <= action_idx < len(_levels) else 0
                if order_qty == 0:
                    return f"No order (stock: {stock:.0f} units)"
                elif order_qty <= 30:
                    return f"Small order of {order_qty} units (stock: {stock:.0f})"
                else:
                    return f"Large order of {order_qty} units (stock: {stock:.0f})"

from custom_tools.warehouse_qa import WarehouseQA

# --- Global action grids (single source of truth) ---
ACTION_LEVELS_BASE = (0, 10, 20, 30, 40, 50, 60, 70, 80)  # baselines
ACTION_LEVELS_RL   = (0, 10, 20, 40, 60, 80)              # matches RL training

# Page config
st.set_page_config(
    page_title="Agentic Inventory Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .highlight {
        background-color: #ffe6e6;
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = WarehouseQA()


def main():
    st.title("🏭 Agentic Inventory Optimizer")
    st.markdown("**Reinforcement Learning for Multi-Warehouse Inventory Management**")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Show RL status
        with st.expander("🤖 RL Model Status", expanded=False):
            if not SB3_AVAILABLE:
                st.error("stable-baselines3 not installed")
                st.code("pip install stable-baselines3")
            elif RLInventoryAgent is None:
                st.warning("RL agent module not found")
                st.info("Create `agents/rl_agent.py` with the provided code")
            else:
                st.success("RL agent module loaded")

                # Check for trained models
                models_dir = "results/models"
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
                    if model_files:
                        st.success(f"Found {len(model_files)} trained models:")
                        for model in model_files:
                            st.text(f"  • {model}")
                    else:
                        st.info("No trained models found")
                else:
                    st.info("Models directory not found")

        # Environment Parameters
        st.subheader("Environment Settings")

        # Fixed at 2 warehouses - just display it
        st.info("📦 **Number of Warehouses:** 2")
        n_warehouses = 2  # Fixed value

        # Cost Parameters
        st.subheader("Cost Parameters")
        st.info("💡 These defaults match the training parameters for optimal model performance")
        holding_cost = st.slider("Holding Cost ($/unit/day)", 0.1, 2.0, 1.0, 0.1)
        stockout_cost = st.slider("Stockout Cost ($/unit)", 1.0, 20.0, 5.0, 0.5)
        order_cost = st.slider("Order Cost ($/order)", 0.1, 5.0, 0.5, 0.1)

        # Demand Parameters
        st.subheader("Demand Settings")
        demand_mean = st.slider("Mean Demand", 10, 100, 30, 5)
        demand_std = st.slider("Demand Volatility (Std Dev)", 5, 30, 15, 5)

        # Lead Time
        lead_time = st.slider("Lead Time (days)", 1, 5, 2, 1)

        # Policy Selection
        st.subheader("Policy Selection")

        # Build policy options with descriptions
        policy_info = {
            "Fixed Order (Baseline 1)": "Always orders fixed quantity",
            "Order-up-to-S (Baseline 2)": "Orders up to target level S=100"
        }

        models_dir = "results/models"
        if os.path.exists(models_dir):
            # DQN models
            dqn_models = sorted([f for f in os.listdir(models_dir) if f.startswith("dqn") and f.endswith(".zip")])
            for model in dqn_models:
                model_name = model[:-4]  # Remove .zip
                key = f"DQN ({model_name})"
                policy_info[key] = "DQN with improvements" if "improved" in model_name else "Deep Q-Network agent"

            # PPO models
            ppo_models = sorted([f for f in os.listdir(models_dir) if f.startswith("ppo") and f.endswith(".zip")])
            for model in ppo_models:
                model_name = model[:-4]  # Remove .zip
                key = f"PPO ({model_name})"
                if "best" in model_name:
                    policy_info[key] = "⭐ Best PPO model"
                elif "balanced" in model_name:
                    policy_info[key] = "Balanced cost/service (92% target)"
                elif "high_service" in model_name:
                    policy_info[key] = "High service level (95% target)"
                else:
                    policy_info[key] = "Proximal Policy Optimization"

                # Add fallback version
                fallback_key = f"PPO ({model_name}) with Fallback"
                policy_info[fallback_key] = policy_info[key] + " + safety fallback"

        # Create selectbox with format_func to show descriptions
        policy_type = st.selectbox(
            "Select Policy",
            list(policy_info.keys()),
            format_func=lambda x: f"{x} - {policy_info.get(x, '')}",
            help="Choose from baselines or trained RL models"
        )

        # Simulation Settings
        st.subheader("Simulation")
        n_episodes = st.number_input("Number of Episodes", 1, 100, 20)
        episode_length = st.number_input("Episode Length", 10, 100, 52)

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        with col2:
            export_results = st.button("💾 Export Results", use_container_width=True)

    # Main Content Area
    tabs = st.tabs(["📊 Simulation", "📈 Analytics", "🔍 Q&A", "⚖️ Comparison", "ℹ️ About"])

    # Tab 1: Simulation
    with tabs[0]:
        if run_sim:
            with st.spinner("Running simulation..."):
                results = run_simulation(
                    n_warehouses, n_episodes, episode_length,
                    holding_cost, stockout_cost, order_cost,
                    demand_mean, demand_std, lead_time,
                    policy_type
                )
                if results is not None:
                    st.session_state.simulation_data = results
                    st.session_state.comparison_results[policy_type] = results['summary']

        if st.session_state.simulation_data:
            display_simulation_results(st.session_state.simulation_data)

    # Tab 2: Analytics
    with tabs[1]:
        if st.session_state.simulation_data:
            display_analytics(st.session_state.simulation_data)
        else:
            st.info("👈 Run a simulation first to see analytics")

    # Tab 3: Q&A System
    with tabs[2]:
        st.subheader("🤖 Warehouse Intelligence Assistant")

        question = st.text_input("Ask a question about warehouse operations:")

        if st.button("Get Answer"):
            if st.session_state.simulation_data:
                answer = st.session_state.qa_system.answer(
                    question,
                    st.session_state.simulation_data.get('rollouts'),
                    st.session_state.comparison_results
                )
                st.markdown(answer)
            else:
                st.warning("Please run a simulation first!")

        # Example questions
        with st.expander("📝 Example Questions"):
            st.markdown("""
            - Which episode had the worst stockouts?
            - What's the average service level?
            - How can we reduce costs?
            - Why are holding costs high?
            - Compare the policies
            """)

    # Tab 4: Comparison
    with tabs[3]:
        if len(st.session_state.comparison_results) >= 2:
            display_comparison(st.session_state.comparison_results)
        else:
            st.info("Run simulations with different policies to compare (minimum 2)")

    # Tab 5: About
    with tabs[4]:
        st.markdown("""
        ## 🎯 Agentic Inventory Optimizer

        This system uses **Reinforcement Learning** to optimize multi-warehouse inventory management,
        achieving significant cost reductions while maintaining high service levels.

        ### 📊 Performance Results

        Using the default cost parameters (Holding: $1.0, Stockout: $5.0, Order: $0.5):

        | Model | Service Level | Cost/Episode | vs Baseline |
        |-------|--------------|--------------|-------------|
        | **DQN Improved** | **98.3%** | **$6,897** | **-42%** |
        | PPO Best | 92-95% | ~$8,000 | -33% |
        | PPO High Service | 86.2% | $11,080 | -7% |
        | Order-up-to-S | 85-90% | ~$12,000 | Baseline |
        | Fixed Order | 70-85% | ~$15,000 | +25% |

        ### 🏆 Key Achievements
        - **42% cost reduction** with DQN vs baseline
        - **98.3% service level** (exceeds 92% target)
        - **Balanced cost structure**: ~66% holding, ~7% stockout, ~27% order

        ### 📈 How It Works
        1. **State Space**: Stock levels, demand history, in-transit orders
        2. **Action Space**: Discrete order quantities (0, 10, 20, 40, 60, 80 units)
        3. **Reward**: Negative total cost with service level shaping
        4. **Training**: 3000-10000 episodes with PPO/DQN algorithms

        ### ⚙️ Important Parameters
        - **Target Service Level**: 92% (configurable)
        - **Below Target Multiplier**: 10x penalty for missing service target
        - **Lead Time**: 1-2 days for order delivery
        - **Demand Pattern**: Seasonal with random spikes (15% chance)

        ### 📝 Usage Tips
        1. Keep cost parameters at defaults for best model performance
        2. Run 20+ episodes for stable statistics
        3. Compare multiple policies to see improvements
        4. Use Q&A system for insights and recommendations

        ### 👥 Project Information
        - **Course**: Reinforcement Learning for Agentic AI Systems
        - **Implementation**: Multi-agent orchestration with safety fallbacks
        - **Tools**: Stable-Baselines3, Gymnasium, Streamlit
        - **Models**: DQN and PPO with VecNormalize
        """)

        # Add performance chart
        st.subheader("📊 Cost Reduction Achievement")

        perf_data = pd.DataFrame({
            'Policy': ['Fixed Order', 'Order-up-to-S', 'PPO High Svc', 'PPO Best', 'DQN Improved'],
            'Cost': [15000, 12000, 11080, 8000, 6897],
            'Service': [0.75, 0.87, 0.862, 0.93, 0.983]
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Cost comparison
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        ax1.bar(perf_data['Policy'], perf_data['Cost'], color=colors)
        ax1.set_ylabel('Cost per Episode ($)')
        ax1.set_title('Cost Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Service level comparison
        ax2.bar(perf_data['Policy'], perf_data['Service'] * 100, color=colors)
        ax2.axhline(92, color='red', linestyle='--', label='Target (92%)')
        ax2.set_ylabel('Service Level (%)')
        ax2.set_title('Service Level Achievement')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)

    # Export functionality
    if export_results and st.session_state.simulation_data:
        exporter = DashboardExporter()
        export_path = exporter.export_all(
            st.session_state.simulation_data['rollouts'],
            st.session_state.simulation_data['summary'],
            policy_type
        )
        st.success(f"✅ Results exported to: {export_path}")


def run_simulation(n_warehouses, n_episodes, episode_length,
                   holding_cost, stockout_cost, order_cost,
                   demand_mean, demand_std, lead_time, policy_type):
    """Run inventory simulation with selected policy."""

    # 1) Choose action grid first (so it's defined for all branches)
    if "DQN" in policy_type or "PPO" in policy_type:
        chosen_levels = ACTION_LEVELS_RL
    else:
        chosen_levels = ACTION_LEVELS_BASE

    # 2) Cost params
    cost_params = CostParams(
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        order_cost=order_cost,
        transfer_cost=0.2  # Default transfer cost
    )

    # 3) Create environment with chosen action grid
    env = InventoryEnv(
        n_warehouses=n_warehouses,
        horizon=episode_length,
        action_levels=chosen_levels,
        cost_params=cost_params,
        demand_mu=(demand_mean, demand_mean),            # same mean for both
        demand_season_amp=(demand_std, demand_std),      # use std as seasonal amp
        lead_times=(lead_time, lead_time),
        target_service=0.92,
        below_target_mult=10.0
    )

    # 4) Select agent
    if policy_type == "Fixed Order (Baseline 1)":
        agent = lambda obs: fixed_order_policy(
            obs, action_levels=chosen_levels, fixed=30, n_warehouses=n_warehouses
        )
        agent_type = "function"

    elif policy_type == "Order-up-to-S (Baseline 2)":
        agent = lambda obs: order_up_to_S(
            obs, S=100, action_levels=chosen_levels, n_warehouses=n_warehouses
        )
        agent_type = "function"

    elif "DQN" in policy_type or "PPO" in policy_type:
        if RLInventoryAgent is None:
            st.error("❌ RL agent module not found. Please create `agents/rl_agent.py`.")
            return None

        # Extract model filename from policy type
        # Example: "PPO (ppo_balanced_92_gain8)" -> "ppo_balanced_92_gain8.zip"
        if "(" in policy_type and ")" in policy_type:
            policy_clean = policy_type.replace(" with Fallback", "")
            model_name = policy_clean.split("(")[1].split(")")[0] + ".zip"
        else:
            algo = "dqn" if "DQN" in policy_type else "ppo"
            model_name = f"{algo}_W{n_warehouses}.zip"

        model_path = f"results/models/{model_name}"

        if not os.path.exists(model_path):
            st.error(f"⚠️ Model not found at `{model_path}` — falling back to Order-up-to-S.")
            agent = lambda obs: order_up_to_S(
                obs, S=100, action_levels=chosen_levels, n_warehouses=n_warehouses
            )
            agent_type = "function"
        else:
            st.success(f"✅ Loading model from `{model_path}`")
            agent = RLInventoryAgent(
                model_path=model_path,
                use_fallback=("Fallback" in policy_type),
                n_warehouses=n_warehouses
            )
            agent_type = "rl"
    else:
        st.error("Unknown policy type.")
        return None

    # 5) Explainer
    try:
        explainer = DecisionExplainer()
    except Exception:
        explainer = None

    # 6) Run episodes
    rollout_data, total_costs, service_levels = [], [], []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        episode_cost = 0.0
        episode_service = []

        for step in range(episode_length):
            # Agent decision
            if agent_type == "function":
                action = agent(obs)               # array of indices
            else:  # "rl"
                action = agent.predict(obs)       # array of indices

            if isinstance(action, (int, np.integer)):
                action = np.array([action] * n_warehouses)
            elif not isinstance(action, np.ndarray):
                action = np.array(action)

            # Convert to qty for logging/explanations (use first WH for explanation)
            a_idx = int(action[0])
            order_qty = chosen_levels[a_idx] if 0 <= a_idx < len(chosen_levels) else 0

            # Explanation (best-effort)
            explanation = ""
            if explainer:
                try:
                    if hasattr(explainer, 'explain'):
                        explanation = explainer.explain(obs, a_idx, env)
                    else:
                        stock = obs[0] if len(obs) > 0 else 0
                        if order_qty == 0:
                            explanation = f"No order (stock: {stock:.0f})"
                        elif order_qty <= 30:
                            explanation = f"Order {order_qty} units (stock: {stock:.0f})"
                        else:
                            explanation = f"Large order {order_qty} units (stock: {stock:.0f})"
                except Exception:
                    explanation = f"Order: {order_qty}"

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_cost_step = info.get('holding_cost', 0.0) + info.get('stockout_cost', 0.0) + info.get('order_cost', 0.0)

            # Record data
            rollout_data.append({
                'episode': episode,
                'step': step,
                'warehouse_1_stock': obs[0] if len(obs) > 0 else 0,
                'warehouse_2_stock': obs[1] if len(obs) > 1 else 0,
                'action_idx': a_idx,
                'order_qty': order_qty,
                'reward': reward,
                'holding_cost': info.get('holding_cost', 0.0),
                'stockout_cost': info.get('stockout_cost', 0.0),
                'order_cost': info.get('order_cost', 0.0),
                'total_cost': total_cost_step,
                'service_level': info.get('service_level', 0.0),
                'explanation': explanation
            })

            episode_cost += total_cost_step
            episode_service.append(info.get('service_level', 0.0))

            obs = next_obs
            if done:
                break

        total_costs.append(episode_cost)
        service_levels.append(np.mean(episode_service))

        progress_bar.progress((episode + 1) / n_episodes)
        status_text.text(f"Episode {episode + 1}/{n_episodes} - Cost: ${episode_cost:.2f}")

    progress_bar.empty()
    status_text.empty()

    # Create summary
    summary = {
        'mean_total_cost': float(np.mean(total_costs)),
        'std_total_cost': float(np.std(total_costs)),
        'mean_service': float(np.mean(service_levels)),
        'std_service': float(np.std(service_levels)),
        'mean_holding': float(np.mean([d['holding_cost'] for d in rollout_data])),
        'mean_stockout': float(np.mean([d['stockout_cost'] for d in rollout_data])),
        'mean_order': float(np.mean([d['order_cost'] for d in rollout_data])),
        'action_levels_used': list(chosen_levels)
    }

    return {
        'rollouts': pd.DataFrame(rollout_data),
        'summary': summary,
        'total_costs': total_costs,
        'service_levels': service_levels
    }


def display_simulation_results(results):
    """Display simulation results."""
    st.subheader("📊 Simulation Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Total Cost",
            f"${results['summary']['mean_total_cost']:.2f}",
            f"±${results['summary']['std_total_cost']:.2f}"
        )

    with col2:
        st.metric(
            "Service Level",
            f"{results['summary']['mean_service']:.1%}",
            f"±{results['summary']['std_service']*100:.1f}%"
        )

    with col3:
        st.metric(
            "Avg Holding Cost",
            f"${results['summary']['mean_holding']:.2f}"
        )

    with col4:
        st.metric(
            "Avg Stockout Cost",
            f"${results['summary']['mean_stockout']:.2f}"
        )

    # Cost breakdown chart
    st.subheader("💰 Cost Breakdown")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  
    # Fix accidental typo above:
    plt.close(fig)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart of costs
    costs = [
        results['summary']['mean_holding'],
        results['summary']['mean_stockout'],
        results['summary']['mean_order']
    ]
    labels = ['Holding', 'Stockout', 'Order']
    ax1.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Cost Distribution")

    # Learning curve
    ax2.plot(results['total_costs'], alpha=0.7)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Cost ($)")
    ax2.set_title("Cost per Episode")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Recent decisions table
    st.subheader("📝 Recent Decisions")
    recent = results['rollouts'].tail(10)[
        ['episode', 'step', 'warehouse_1_stock', 'order_qty', 'total_cost', 'service_level', 'explanation']
    ]
    st.dataframe(recent, use_container_width=True)


def display_analytics(results):
    """Display detailed analytics."""
    st.subheader("📈 Detailed Analytics")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Inventory levels over time
    df = results['rollouts']
    axes[0, 0].plot(df['warehouse_1_stock'].rolling(10).mean(), label='Warehouse 1', alpha=0.7)
    if 'warehouse_2_stock' in df.columns:
        axes[0, 0].plot(df['warehouse_2_stock'].rolling(10).mean(), label='Warehouse 2', alpha=0.7)
    axes[0, 0].set_title("Inventory Levels (10-step MA)")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Stock Level")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Service level distribution
    axes[0, 1].hist(results['service_levels'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.92, color='r', linestyle='--', label='Target (92%)')
    axes[0, 1].set_title("Service Level Distribution")
    axes[0, 1].set_xlabel("Service Level")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Cost components over episodes
    episode_costs = df.groupby('episode').agg({
        'holding_cost': 'sum',
        'stockout_cost': 'sum',
        'order_cost': 'sum'
    })
    axes[1, 0].plot(episode_costs.index, episode_costs['holding_cost'], label='Holding', alpha=0.7)
    axes[1, 0].plot(episode_costs.index, episode_costs['stockout_cost'], label='Stockout', alpha=0.7)
    axes[1, 0].plot(episode_costs.index, episode_costs['order_cost'], label='Order', alpha=0.7)
    axes[1, 0].set_title("Cost Components by Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cost ($)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Order quantity distribution
    qty_counts = df['order_qty'].value_counts().sort_index()
    axes[1, 1].bar(qty_counts.index, qty_counts.values, alpha=0.7)
    axes[1, 1].set_title("Order Quantity Distribution")
    axes[1, 1].set_xlabel("Order Qty")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Statistics table
    st.subheader("📊 Statistical Summary")

    stats_df = pd.DataFrame({
        'Metric': ['Total Cost', 'Service Level', 'Holding Cost', 'Stockout Cost', 'Order Cost'],
        'Mean': [
            results['summary']['mean_total_cost'],
            results['summary']['mean_service'],
            results['summary']['mean_holding'],
            results['summary']['mean_stockout'],
            results['summary']['mean_order']
        ],
        'Std Dev': [
            results['summary']['std_total_cost'],
            results['summary']['std_service'],
            df['holding_cost'].std(),
            df['stockout_cost'].std(),
            df['order_cost'].std()
        ],
        'Min': [
            min(results['total_costs']),
            min(results['service_levels']),
            df['holding_cost'].min(),
            df['stockout_cost'].min(),
            df['order_cost'].min()
        ],
        'Max': [
            max(results['total_costs']),
            max(results['service_levels']),
            df['holding_cost'].max(),
            df['stockout_cost'].max(),
            df['order_cost'].max()
        ]
    })

    st.dataframe(stats_df, use_container_width=True)


def display_comparison(comparison_results):
    """Display policy comparison."""
    st.subheader("⚖️ Policy Comparison")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df = comparison_df.round(2)

    # Display table
    st.dataframe(comparison_df, use_container_width=True)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cost comparison
    policies = list(comparison_results.keys())
    costs = [comparison_results[p]['mean_total_cost'] for p in policies]
    axes[0].bar(policies, costs, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple'][:len(policies)])
    axes[0].set_title("Mean Total Cost by Policy")
    axes[0].set_ylabel("Cost ($)")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Service level comparison
    services = [comparison_results[p]['mean_service'] for p in policies]
    axes[1].bar(policies, services, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple'][:len(policies)])
    axes[1].axhline(0.92, color='black', linestyle='--', label='Target')
    axes[1].set_title("Mean Service Level by Policy")
    axes[1].set_ylabel("Service Level")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Best policy recommendation
    best_policy = min(comparison_results.items(), key=lambda x: x[1]['mean_total_cost'])
    st.success(f"🏆 **Recommended Policy:** {best_policy[0]} with ${best_policy[1]['mean_total_cost']:.2f} average cost")


if __name__ == "__main__":
    main()
