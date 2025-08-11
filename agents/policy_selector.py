# agents/policy_selector.py
import numpy as np
import math
from typing import Dict, List, Optional

class UCBBandit:
    """Upper Confidence Bound bandit for policy selection."""
    
    def __init__(self, arms: List[str], c: float = 1.4):
        """
        Initialize UCB bandit.
        
        Args:
            arms: List of policy names ["ppo", "dqn", "heuristic"]
            c: Exploration constant
        """
        self.arms = arms
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}
        self.c = c
        self.t = 0
        self.current_arm = None
        
    def select(self) -> str:
        """Select next arm using UCB algorithm."""
        self.t += 1
        
        # Play untried arms first
        for arm in self.arms:
            if self.counts[arm] == 0:
                self.current_arm = arm
                return arm
                
        # UCB score calculation
        def ucb(arm):
            exploitation = self.values[arm]
            exploration = self.c * math.sqrt(math.log(self.t) / self.counts[arm])
            return exploitation + exploration
            
        self.current_arm = max(self.arms, key=ucb)
        return self.current_arm
        
    def update(self, reward: float):
        """
        Update arm statistics.
        
        Args:
            reward: Reward for the last selected arm (use -cost for cost minimization)
        """
        if self.current_arm is None:
            return
            
        arm = self.current_arm
        self.counts[arm] += 1
        n = self.counts[arm]
        v = self.values[arm]
        # Incremental average update
        self.values[arm] = v + (reward - v) / n
        
    def get_stats(self) -> Dict:
        """Get current bandit statistics."""
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_pulls": self.t,
            "current_arm": self.current_arm,
            "best_arm": max(self.values, key=self.values.get) if self.values else None
        }
        
    def reset(self):
        """Reset bandit statistics."""
        self.counts = {a: 0 for a in self.arms}
        self.values = {a: 0.0 for a in self.arms}
        self.t = 0
        self.current_arm = None