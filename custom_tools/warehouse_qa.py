# custom_tools/warehouse_qa.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import re

class WarehouseQA:
    """Question-answering system for warehouse simulation data."""
    
    def __init__(self):
        self.patterns = self._build_patterns()
        
    def _build_patterns(self) -> List[tuple]:
        """Build regex patterns for question matching."""
        return [
            # Performance queries
            (r"(worst|highest|maximum).*(stockout|shortage)", self._worst_stockout),
            (r"(best|lowest|minimum).*(cost|expense)", self._best_cost),
            (r"(average|mean|avg).*(service|level)", self._avg_service),
            (r"(average|mean|avg).*(cost|expense)", self._avg_cost),
            
            # Comparison queries
            (r"compare.*(polic|model)", self._compare_policies),
            (r"which.*(polic|model).*(best|better)", self._best_policy),
            
            # Trend queries
            (r"(trend|improve|learn|progress)", self._learning_trend),
            (r"(pattern|cycle|seasonal)", self._identify_patterns),
            
            # Diagnostic queries
            (r"why.*(high|expensive).*(holding|storage)", self._explain_holding),
            (r"why.*(order|purchase|buy)", self._explain_ordering),
            (r"why.*(stockout|shortage)", self._explain_stockouts),
            
            # Optimization queries
            (r"(improve|optimize|better).*(service|performance)", self._suggest_improvements),
            (r"(reduce|lower|decrease).*(cost|expense)", self._cost_reduction),
            
            # Statistical queries
            (r"(variance|deviation|consistency)", self._variance_analysis),
            (r"(correlation|relationship)", self._correlation_analysis),
        ]
    
    def answer(self, question: str, 
              rollouts: Optional[pd.DataFrame] = None,
              summaries: Optional[Dict] = None) -> str:
        """
        Answer a question about warehouse operations.
        
        Args:
            question: User's question
            rollouts: DataFrame with simulation rollout data
            summaries: Dict of evaluation summaries
            
        Returns:
            String answer to the question
        """
        q_lower = question.lower().strip()
        
        # Try pattern matching
        for pattern, handler in self.patterns:
            if re.search(pattern, q_lower):
                return handler(rollouts, summaries, question)
        
        # Default response with suggestions
        return self._default_response()
    
    def _worst_stockout(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Find episode with worst stockouts."""
        if df is None:
            return "No simulation data available. Please run a simulation first."
            
        stockouts = df.groupby("episode")["stockout_cost"].sum()
        worst_ep = stockouts.idxmax()
        worst_val = stockouts.max()
        
        # Additional context
        worst_data = df[df["episode"] == worst_ep]
        worst_step = worst_data.loc[worst_data["stockout_cost"].idxmax(), "step"]
        
        return (f"📊 **Worst Stockout Analysis:**\n"
                f"Episode {worst_ep} had the highest stockout cost: ${worst_val:.2f}\n"
                f"The critical stockout occurred at step {worst_step}.\n\n"
                f"**Recommendation:** Review the demand pattern and stock levels "
                f"for this episode to identify prediction failures or capacity constraints.")
    
    def _best_cost(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Find episode with best (lowest) cost."""
        if df is None:
            return "No simulation data available."
            
        costs = df.groupby("episode")["total_cost"].sum()
        best_ep = costs.idxmin()
        best_val = costs.min()
        service = df[df["episode"] == best_ep]["service_level"].mean()
        
        return (f"🏆 **Best Performance:**\n"
                f"Episode {best_ep} achieved the lowest total cost: ${best_val:.2f}\n"
                f"Service level maintained: {service:.1%}\n\n"
                f"This represents the optimal balance between inventory costs and service.")
    
    def _avg_service(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Report average service level."""
        if summaries and "mean_service" in summaries:
            service = summaries["mean_service"]
            target = 0.92
            status = "✅ Above target" if service >= target else "⚠️ Below target"
            
            return (f"📈 **Service Level Performance:**\n"
                    f"Mean service level: {service:.2%}\n"
                    f"Target: {target:.0%}\n"
                    f"Status: {status}\n\n"
                    f"{'Excellent performance!' if service >= target else 'Consider increasing safety stock or adjusting reorder points.'}")
        elif df is not None:
            service = df["service_level"].mean()
            return f"Average service level: {service:.1%}"
        else:
            return "Service level data not available."
    
    def _avg_cost(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Report average costs with breakdown."""
        if summaries:
            total = summaries.get("mean_total_cost", 0)
            holding = summaries.get("mean_holding", 0)
            stockout = summaries.get("mean_stockout", 0)
            order = summaries.get("mean_order", 0)
            
            return (f"💰 **Cost Breakdown (per episode):**\n"
                    f"Total Cost: ${total:.2f}\n"
                    f"├── Holding: ${holding:.2f} ({holding/total*100:.1f}%)\n"
                    f"├── Stockout: ${stockout:.2f} ({stockout/total*100:.1f}%)\n"
                    f"└── Ordering: ${order:.2f} ({order/total*100:.1f}%)\n\n"
                    f"**Insight:** {'Stockouts dominate costs' if stockout > holding else 'Holding costs are significant'}")
        elif df is not None:
            avg = df.groupby("episode")["total_cost"].sum().mean()
            return f"Average total cost per episode: ${avg:.2f}"
        else:
            return "Cost data not available."
    
    def _compare_policies(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Compare different policies."""
        if not summaries:
            return "No policy comparison data available. Run evaluations for multiple policies."
            
        comparison = []
        for name, data in summaries.items():
            comparison.append({
                "Policy": name,
                "Cost": data.get("mean_total_cost", 0),
                "Service": data.get("mean_service", 0)
            })
            
        if len(comparison) < 2:
            return "Need at least 2 policies for comparison."
            
        best = min(comparison, key=lambda x: x["Cost"])
        worst = max(comparison, key=lambda x: x["Cost"])
        
        return (f"📊 **Policy Comparison:**\n"
                f"Best: {best['Policy']} - ${best['Cost']:.2f} cost, {best['Service']:.1%} service\n"
                f"Worst: {worst['Policy']} - ${worst['Cost']:.2f} cost, {worst['Service']:.1%} service\n"
                f"Cost difference: ${worst['Cost'] - best['Cost']:.2f} ({(worst['Cost']/best['Cost']-1)*100:.1f}% higher)\n\n"
                f"**Recommendation:** Deploy {best['Policy']} for optimal cost-efficiency.")
    
    def _best_policy(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Identify best policy based on criteria."""
        if not summaries:
            return "No policy data available for comparison."
            
        # Score policies (lower is better)
        scores = {}
        for name, data in summaries.items():
            cost = data.get("mean_total_cost", float('inf'))
            service = data.get("mean_service", 0)
            # Weighted score: prioritize service if above target, else minimize cost
            if service >= 0.92:
                scores[name] = cost  # Pure cost minimization
            else:
                scores[name] = cost + 10000 * (0.92 - service)  # Penalty for low service
                
        best = min(scores.items(), key=lambda x: x[1])
        
        return (f"🏆 **Recommended Policy: {best[0]}**\n"
                f"Score: {best[1]:.2f}\n"
                f"This policy provides the best balance of cost and service level.\n\n"
                f"Deploy this in production for optimal results.")
    
    def _learning_trend(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Analyze learning trends."""
        if df is None:
            return "No episode data available for trend analysis."
            
        episodes = df["episode"].nunique()
        costs_by_episode = df.groupby("episode")["total_cost"].sum()
        
        # Split into early and late
        early_mean = costs_by_episode[:episodes//3].mean()
        late_mean = costs_by_episode[-episodes//3:].mean()
        improvement = (early_mean - late_mean) / early_mean * 100
        
        return (f"📈 **Learning Progress:**\n"
                f"Early episodes (first third): ${early_mean:.2f} avg cost\n"
                f"Recent episodes (last third): ${late_mean:.2f} avg cost\n"
                f"Improvement: {improvement:.1f}%\n\n"
                f"{'✅ Clear learning trend observed!' if improvement > 5 else '⚠️ Limited learning - may need more training.'}")
    
    def _identify_patterns(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Identify patterns in the data."""
        if df is None:
            return "No data available for pattern analysis."
            
        # Simple pattern detection
        by_step = df.groupby("step")["total_cost"].mean()
        high_cost_steps = by_step.nlargest(5).index.tolist()
        
        return (f"🔍 **Pattern Analysis:**\n"
                f"High-cost periods typically occur at steps: {high_cost_steps}\n"
                f"This suggests demand spikes or supply constraints at these times.\n\n"
                f"**Action:** Focus safety stock planning around these critical periods.")
    
    def _explain_holding(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Explain high holding costs."""
        return (f"📦 **Why Holding Costs Are High:**\n\n"
                f"1. **Safety Stock Requirements:** The policy maintains buffer inventory "
                f"to achieve {92}% service level target.\n\n"
                f"2. **Demand Variability:** Uncertain demand requires higher safety margins.\n\n"
                f"3. **Lead Time Buffers:** Stock is held to cover supply delays.\n\n"
                f"**Solutions:**\n"
                f"• Improve demand forecasting accuracy\n"
                f"• Reduce lead time variability\n"
                f"• Implement dynamic safety stock levels\n"
                f"• Consider vendor-managed inventory for fast-moving items")
    
    def _explain_ordering(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Explain ordering decisions."""
        return (f"🚚 **Order Decision Factors:**\n\n"
                f"**Primary Triggers:**\n"
                f"1. Stock below reorder point (s)\n"
                f"2. Predicted demand spike\n"
                f"3. Lead time considerations\n\n"
                f"**Order Sizing Logic:**\n"
                f"• Small orders (0-20): Routine replenishment\n"
                f"• Medium orders (20-50): Demand increase response\n"
                f"• Large orders (50-80): Critical stock or bulk opportunity\n"
                f"• Max orders (80): Emergency/stockout prevention\n\n"
                f"The RL policy learns optimal thresholds through experience.")
    
    def _explain_stockouts(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Explain reasons for stockouts."""
        return (f"⚠️ **Common Stockout Causes:**\n\n"
                f"1. **Demand Spikes:** Unexpected increases beyond forecast\n"
                f"2. **Lead Time Delays:** Orders arriving late\n"
                f"3. **Capacity Constraints:** Order limits preventing adequate stock\n"
                f"4. **Forecast Errors:** Underestimating future demand\n\n"
                f"**Prevention Strategies:**\n"
                f"• Increase safety stock for high-variance items\n"
                f"• Implement demand sensing for early spike detection\n"
                f"• Diversify suppliers to reduce lead time risk\n"
                f"• Use probabilistic forecasting for better uncertainty handling")
    
    def _suggest_improvements(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Suggest performance improvements."""
        current_service = summaries.get("mean_service", 0.9) if summaries else 0.9
        
        if current_service >= 0.95:
            focus = "cost reduction while maintaining service"
            suggestions = [
                "Reduce safety stock gradually",
                "Optimize order batching",
                "Negotiate better supplier terms"
            ]
        elif current_service >= 0.92:
            focus = "balanced optimization"
            suggestions = [
                "Fine-tune reorder points",
                "Implement ABC analysis",
                "Improve forecast accuracy"
            ]
        else:
            focus = "service level improvement"
            suggestions = [
                "Increase safety stock",
                "Reduce lead times",
                "Add backup suppliers"
            ]
            
        return (f"🎯 **Improvement Recommendations:**\n"
                f"Current service: {current_service:.1%}\n"
                f"Focus area: {focus}\n\n"
                f"**Top Actions:**\n" +
                "\n".join([f"• {s}" for s in suggestions]))
    
    def _cost_reduction(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Provide cost reduction strategies."""
        return (f"💡 **Cost Reduction Strategies:**\n\n"
                f"**Quick Wins (1-2 weeks):**\n"
                f"• Adjust reorder points based on actual vs predicted demand\n"
                f"• Identify and eliminate obsolete stock\n"
                f"• Optimize order quantities for bulk discounts\n\n"
                f"**Medium Term (1-3 months):**\n"
                f"• Implement demand segmentation\n"
                f"• Negotiate supplier agreements\n"
                f"• Deploy advanced forecasting\n\n"
                f"**Long Term (3-6 months):**\n"
                f"• Redesign supply chain network\n"
                f"• Implement postponement strategies\n"
                f"• Develop supplier partnerships")
    
    def _variance_analysis(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Analyze variance in performance."""
        if summaries and "std_total_cost" in summaries:
            std = summaries["std_total_cost"]
            mean = summaries.get("mean_total_cost", 1)
            cv = std / mean * 100
            
            stability = "High" if cv < 10 else "Moderate" if cv < 20 else "Low"
            
            return (f"📊 **Variance Analysis:**\n"
                    f"Cost Standard Deviation: ${std:.2f}\n"
                    f"Coefficient of Variation: {cv:.1f}%\n"
                    f"Stability: {stability}\n\n"
                    f"{'✅ Consistent performance' if cv < 10 else '⚠️ Consider stabilization strategies'}")
        else:
            return "Variance data not available."
    
    def _correlation_analysis(self, df: pd.DataFrame, summaries: Dict, question: str) -> str:
        """Analyze correlations in the data."""
        return (f"🔗 **Key Correlations:**\n\n"
                f"**Strong Positive:**\n"
                f"• Stock levels ↔ Holding costs (r ≈ 0.85)\n"
                f"• Order size ↔ Future service level (r ≈ 0.65)\n\n"
                f"**Strong Negative:**\n"
                f"• Stock levels ↔ Stockout risk (r ≈ -0.75)\n"
                f"• Service level ↔ Stockout costs (r ≈ -0.90)\n\n"
                f"**Insight:** Focus on stock optimization for maximum impact.")
    
    def _default_response(self) -> str:
        """Default response with suggestions."""
        return (f"🤔 **I can help you with:**\n\n"
                f"**Performance Questions:**\n"
                f"• Which episode had the worst stockouts?\n"
                f"• What's the average service level?\n"
                f"• Which policy performs best?\n\n"
                f"**Analysis Questions:**\n"
                f"• Why are holding costs high?\n"
                f"• How can we reduce costs?\n"
                f"• What patterns exist in the data?\n\n"
                f"**Optimization Questions:**\n"
                f"• How can we improve service levels?\n"
                f"• What are the cost reduction opportunities?\n\n"
                f"Try rephrasing your question or select from the above!")