# custom_tools/dashboard_export_tool.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class DashboardExporter:
    """Tool for exporting dashboard data and visualizations."""
    
    def __init__(self, output_dir: str = "results/exports"):
        """
        Initialize exporter with output directory.
        
        Args:
            output_dir: Directory to save exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_all(self, rollouts: pd.DataFrame, summary: Dict, policy_name: str) -> str:
        """
        Export all data and visualizations.
        
        Args:
            rollouts: DataFrame with simulation data
            summary: Summary statistics dict
            policy_name: Name of the policy
            
        Returns:
            Path to export directory
        """
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.output_dir / f"{policy_name.replace(' ', '_')}_{timestamp}"
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export data
        self.export_csv(rollouts, export_path / "rollout_data.csv")
        self.export_summary_json(summary, export_path / "summary.json")
        
        # Generate and save visualizations
        self.export_cost_breakdown_chart(rollouts, export_path / "cost_breakdown.png")
        self.export_learning_curve(rollouts, export_path / "learning_curve.png")
        self.export_inventory_levels(rollouts, export_path / "inventory_levels.png")
        self.export_service_level_histogram(rollouts, export_path / "service_levels.png")
        
        # Create report
        self.generate_report(rollouts, summary, policy_name, export_path / "report.txt")
        
        return str(export_path)
    
    def export_csv(self, data: pd.DataFrame, filepath: Path):
        """Export DataFrame to CSV."""
        data.to_csv(filepath, index=False)
        print(f"✓ Data exported to {filepath}")
    
    def export_summary_json(self, summary: Dict, filepath: Path):
        """Export summary statistics to JSON."""
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Summary exported to {filepath}")
    
    def export_cost_breakdown_chart(self, data: pd.DataFrame, filepath: Path):
        """Generate and save cost breakdown chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Aggregate costs
        total_holding = data['holding_cost'].sum()
        total_stockout = data['stockout_cost'].sum()
        total_order = data['order_cost'].sum()
        
        # Pie chart
        costs = [total_holding, total_stockout, total_order]
        labels = ['Holding', 'Stockout', 'Order']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax1.pie(costs, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Cost Distribution')
        
        # Bar chart by episode
        episode_costs = data.groupby('episode').agg({
            'holding_cost': 'sum',
            'stockout_cost': 'sum',
            'order_cost': 'sum'
        })
        
        x = np.arange(len(episode_costs))
        width = 0.25
        
        ax2.bar(x - width, episode_costs['holding_cost'], width, label='Holding', color=colors[0])
        ax2.bar(x, episode_costs['stockout_cost'], width, label='Stockout', color=colors[1])
        ax2.bar(x + width, episode_costs['order_cost'], width, label='Order', color=colors[2])
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost ($)')
        ax2.set_title('Cost Components by Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Cost breakdown chart saved to {filepath}")
    
    def export_learning_curve(self, data: pd.DataFrame, filepath: Path):
        """Generate and save learning curve."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episode_costs = data.groupby('episode')['total_cost'].sum()
        episodes = episode_costs.index
        costs = episode_costs.values
        
        # Plot with rolling average
        ax.plot(episodes, costs, alpha=0.5, label='Episode Cost')
        
        if len(costs) > 5:
            rolling_avg = pd.Series(costs).rolling(window=5, center=True).mean()
            ax.plot(episodes, rolling_avg, linewidth=2, label='5-Episode Moving Average')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Learning Curve: Cost Reduction Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(episodes) > 1:
            z = np.polyfit(episodes, costs, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "r--", alpha=0.5, label=f'Trend: ${z[0]:.2f} per episode')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Learning curve saved to {filepath}")
    
    def export_inventory_levels(self, data: pd.DataFrame, filepath: Path):
        """Generate and save inventory level chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot inventory levels
        ax.plot(data.index, data['warehouse_1_stock'], label='Warehouse 1', alpha=0.7)
        if 'warehouse_2_stock' in data.columns:
            ax.plot(data.index, data['warehouse_2_stock'], label='Warehouse 2', alpha=0.7)
        
        # Add stockout markers
        stockout_points = data[data['stockout_cost'] > 0]
        if not stockout_points.empty:
            ax.scatter(stockout_points.index, 
                      stockout_points['warehouse_1_stock'], 
                      color='red', s=50, alpha=0.7, label='Stockout Events')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Inventory Level')
        ax.set_title('Inventory Levels Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for average
        avg_stock = data['warehouse_1_stock'].mean()
        ax.axhline(y=avg_stock, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Average: {avg_stock:.1f}')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Inventory levels chart saved to {filepath}")
    
    def export_service_level_histogram(self, data: pd.DataFrame, filepath: Path):
        """Generate and save service level histogram."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        service_levels = data.groupby('episode')['service_level'].mean()
        
        ax.hist(service_levels, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(0.92, color='red', linestyle='--', linewidth=2, label='Target (92%)')
        ax.axvline(service_levels.mean(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean ({service_levels.mean():.1%})')
        
        ax.set_xlabel('Service Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Service Level Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Service level histogram saved to {filepath}")
    
    def generate_report(self, data: pd.DataFrame, summary: Dict, 
                       policy_name: str, filepath: Path):
        """Generate text report with key metrics and insights."""
        report = []
        report.append("=" * 60)
        report.append(f"INVENTORY OPTIMIZATION REPORT")
        report.append(f"Policy: {policy_name}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Mean Total Cost: ${summary.get('mean_total_cost', 0):.2f}")
        report.append(f"Std Total Cost: ${summary.get('std_total_cost', 0):.2f}")
        report.append(f"Mean Service Level: {summary.get('mean_service', 0):.1%}")
        report.append(f"Mean Holding Cost: ${summary.get('mean_holding', 0):.2f}")
        report.append(f"Mean Stockout Cost: ${summary.get('mean_stockout', 0):.2f}")
        report.append(f"Mean Order Cost: ${summary.get('mean_order', 0):.2f}")
        report.append("")
        
        # Episode analysis
        episode_costs = data.groupby('episode')['total_cost'].sum()
        report.append("EPISODE ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total Episodes: {len(episode_costs)}")
        report.append(f"Best Episode: {episode_costs.idxmin()} (${episode_costs.min():.2f})")
        report.append(f"Worst Episode: {episode_costs.idxmax()} (${episode_costs.max():.2f})")
        
        # Learning progress
        if len(episode_costs) > 3:
            early = episode_costs[:3].mean()
            late = episode_costs[-3:].mean()
            improvement = (early - late) / early * 100
            report.append(f"Learning Progress: {improvement:.1f}% improvement")
        report.append("")
        
        # Cost breakdown
        total_costs = data['total_cost'].sum()
        holding_pct = data['holding_cost'].sum() / total_costs * 100
        stockout_pct = data['stockout_cost'].sum() / total_costs * 100
        order_pct = data['order_cost'].sum() / total_costs * 100
        
        report.append("COST BREAKDOWN")
        report.append("-" * 30)
        report.append(f"Holding: {holding_pct:.1f}%")
        report.append(f"Stockout: {stockout_pct:.1f}%")
        report.append(f"Order: {order_pct:.1f}%")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        if summary.get('mean_service', 0) < 0.92:
            report.append("⚠ Service level below target - consider increasing safety stock")
        else:
            report.append("✓ Service level meets target")
            
        if stockout_pct > 30:
            report.append("⚠ High stockout costs - review reorder points")
        elif holding_pct > 50:
            report.append("⚠ High holding costs - consider reducing safety stock")
        else:
            report.append("✓ Cost distribution is balanced")
        
        # Write report
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Report generated at {filepath}")
        
        return '\n'.join(report)