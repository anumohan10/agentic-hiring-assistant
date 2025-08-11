# custom_tools/decision_explainer_tool.py
import numpy as np
from typing import Dict, List, Optional, Tuple

class DecisionExplainerTool:
    """Explains inventory decisions made by policies."""
    
    def __init__(self, 
                 target_service_level: float = 0.92,
                 stockout_cost: float = 10.0,
                 holding_cost: float = 1.0):
        self.target_service = target_service_level
        self.stockout_cost = stockout_cost
        self.holding_cost = holding_cost
        
    def explain_decision(self, 
                        state: np.ndarray,
                        action: float,
                        policy_name: str = "RL Policy",
                        warehouse_id: int = 0,
                        forecast: Optional[float] = None) -> Dict:
        """
        Generate detailed explanation for an inventory decision.
        
        Args:
            state: Current environment state
            action: Action taken (order quantity)
            policy_name: Name of policy that made decision
            warehouse_id: Which warehouse this decision is for
            forecast: Optional demand forecast
            
        Returns:
            Dict with explanation components
        """
        # Parse state (assuming [stock_w1, stock_w2, demand_lag1, ...])
        n_warehouses = 2
        current_stock = state[warehouse_id] if warehouse_id < n_warehouses else state[0]
        recent_demands = state[n_warehouses:n_warehouses+3] if len(state) > n_warehouses else [50, 50, 50]
        avg_demand = np.mean(recent_demands)
        
        # Calculate key metrics
        days_of_stock = current_stock / avg_demand if avg_demand > 0 else float('inf')
        order_size_category = self._categorize_order(action, avg_demand)
        risk_level = self._assess_risk(current_stock, avg_demand, forecast)
        
        # Build explanation
        explanation = {
            "summary": self._generate_summary(action, order_size_category, risk_level),
            "context": {
                "current_stock": current_stock,
                "avg_recent_demand": avg_demand,
                "days_of_stock_remaining": days_of_stock,
                "forecast": forecast or avg_demand,
                "warehouse_id": warehouse_id,
                "policy": policy_name
            },
            "decision_factors": self._identify_factors(
                current_stock, avg_demand, action, risk_level
            ),
            "rationale": self._generate_rationale(
                action, current_stock, avg_demand, risk_level, order_size_category
            ),
            "risk_assessment": risk_level,
            "cost_implications": self._estimate_cost_impact(
                action, current_stock, avg_demand
            ),
            "confidence": self._assess_confidence(policy_name, risk_level)
        }
        
        return explanation
    
    def _categorize_order(self, action: float, avg_demand: float) -> str:
        """Categorize order size relative to demand."""
        if action == 0:
            return "no_order"
        elif action < avg_demand * 0.5:
            return "minimal"
        elif action < avg_demand:
            return "small"
        elif action < avg_demand * 1.5:
            return "moderate"
        elif action < avg_demand * 2:
            return "large"
        else:
            return "emergency"
    
    def _assess_risk(self, stock: float, demand: float, 
                    forecast: Optional[float] = None) -> Dict:
        """Assess stockout risk level."""
        expected_demand = forecast if forecast else demand
        days_coverage = stock / expected_demand if expected_demand > 0 else float('inf')
        
        if days_coverage < 1:
            risk_level = "critical"
            probability = 0.9
        elif days_coverage < 2:
            risk_level = "high"
            probability = 0.6
        elif days_coverage < 3:
            risk_level = "moderate"
            probability = 0.3
        elif days_coverage < 5:
            risk_level = "low"
            probability = 0.1
        else:
            risk_level = "minimal"
            probability = 0.05
            
        return {
            "level": risk_level,
            "stockout_probability": probability,
            "days_coverage": days_coverage
        }
    
    def _identify_factors(self, stock: float, demand: float, 
                         action: float, risk: Dict) -> List[str]:
        """Identify key decision factors."""
        factors = []
        
        if risk["level"] in ["critical", "high"]:
            factors.append("High stockout risk requiring immediate action")
            
        if stock < demand:
            factors.append("Current stock below average daily demand")
            
        if action > demand * 1.5:
            factors.append("Building safety buffer for demand variability")
            
        if action == 0 and stock > demand * 3:
            factors.append("Sufficient inventory to avoid holding costs")
            
        if self.stockout_cost / self.holding_cost > 5:
            factors.append("High stockout penalty favors maintaining inventory")
            
        return factors
    
    def _generate_summary(self, action: float, category: str, risk: Dict) -> str:
        """Generate one-line summary of decision."""
        if category == "no_order":
            return f"No replenishment needed - {risk['level']} stockout risk with {risk['days_coverage']:.1f} days of coverage"
        elif category == "emergency":
            return f"Emergency order of {action:.0f} units to prevent imminent stockout"
        elif category in ["large", "moderate"]:
            return f"{category.capitalize()} order of {action:.0f} units to maintain service level"
        else:
            return f"Small order of {action:.0f} units for routine replenishment"
    
    def _generate_rationale(self, action: float, stock: float, 
                           demand: float, risk: Dict, category: str) -> str:
        """Generate detailed rationale for the decision."""
        rationale = f"The {category.replace('_', ' ')} order of {action:.0f} units "
        
        if risk["level"] == "critical":
            rationale += "is critical to prevent immediate stockouts. "
            rationale += f"With only {stock:.0f} units and average demand of {demand:.1f}, "
            rationale += "service levels are at severe risk without immediate replenishment."
            
        elif risk["level"] == "high":
            rationale += "addresses the elevated stockout risk. "
            rationale += f"Current inventory provides only {risk['days_coverage']:.1f} days of coverage, "
            rationale += "requiring proactive ordering to maintain target service levels."
            
        elif category == "no_order":
            rationale = f"No order is placed because current stock of {stock:.0f} units "
            rationale += f"provides {risk['days_coverage']:.1f} days of coverage. "
            rationale += "This minimizes holding costs while maintaining acceptable service levels."
            
        else:
            rationale += "balances service requirements with inventory costs. "
            rationale += f"This maintains approximately {risk['days_coverage'] + action/demand:.1f} days "
            rationale += "of forward coverage after replenishment."
            
        # Add cost consideration
        if self.stockout_cost > self.holding_cost * 5:
            rationale += f" Given the {self.stockout_cost/self.holding_cost:.1f}x ratio "
            rationale += "of stockout to holding costs, the policy favors availability over minimizing inventory."
            
        return rationale
    
    def _estimate_cost_impact(self, action: float, stock: float, 
                             demand: float) -> Dict:
        """Estimate cost implications of the decision."""
        # Estimate costs
        immediate_order_cost = action * 2.0  # Assuming $2 per unit ordered
        expected_holding = (stock + action/2) * self.holding_cost
        
        # Estimate stockout risk cost
        if stock < demand:
            expected_stockout = (demand - stock) * self.stockout_cost * 0.5
        else:
            expected_stockout = 0
            
        return {
            "immediate_order_cost": immediate_order_cost,
            "expected_holding_cost": expected_holding,
            "expected_stockout_cost": expected_stockout,
            "total_expected": immediate_order_cost + expected_holding + expected_stockout
        }
    
    def _assess_confidence(self, policy_name: str, risk: Dict) -> str:
        """Assess confidence in the decision."""
        if "PPO" in policy_name or "DQN" in policy_name:
            if risk["level"] in ["critical", "high"]:
                return "High - RL policy trained on similar critical scenarios"
            else:
                return "High - Within normal operating parameters"
        elif "heuristic" in policy_name.lower():
            return "Moderate - Rule-based decision without learning"
        elif "safety" in policy_name.lower():
            return "High - Conservative fallback to ensure service"
        else:
            return "Moderate"
    
    def format_explanation(self, explanation: Dict) -> str:
        """Format explanation for display."""
        output = []
        output.append(f"📊 **Decision Summary**")
        output.append(f"{explanation['summary']}\n")
        
        output.append(f"**Context:**")
        ctx = explanation['context']
        output.append(f"- Warehouse {ctx['warehouse_id']}: {ctx['current_stock']:.0f} units in stock")
        output.append(f"- Recent demand average: {ctx['avg_recent_demand']:.1f} units/day")
        output.append(f"- Days of coverage: {ctx['days_of_stock_remaining']:.1f}")
        output.append(f"- Policy: {ctx['policy']}\n")
        
        output.append(f"**Key Factors:**")
        for factor in explanation['decision_factors']:
            output.append(f"- {factor}")
        output.append("")
        
        output.append(f"**Rationale:**")
        output.append(explanation['rationale'] + "\n")
        
        output.append(f"**Cost Analysis:**")
        costs = explanation['cost_implications']
        output.append(f"- Order cost: ${costs['immediate_order_cost']:.2f}")
        output.append(f"- Expected holding: ${costs['expected_holding_cost']:.2f}")
        output.append(f"- Stockout risk: ${costs['expected_stockout_cost']:.2f}")
        output.append(f"- **Total expected: ${costs['total_expected']:.2f}**\n")
        
        output.append(f"**Risk Level:** {explanation['risk_assessment']['level'].upper()}")
        output.append(f"**Decision Confidence:** {explanation['confidence']}")
        
        return "\n".join(output)