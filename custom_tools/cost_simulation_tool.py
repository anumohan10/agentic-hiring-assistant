"""
Cost tool used by the environment & UI.

holding_cost: cost per unit remaining in stock after serving demand.
stockout_cost: cost per unit of unmet demand.
order_cost: cost per unit ordered (placed this step).
transfer_cost: cost per unit transferred.
"""
from dataclasses import dataclass

@dataclass
class CostParams:
    holding_cost: float = 1.0
    stockout_cost: float = 5.0
    order_cost: float = 0.5
    transfer_cost: float = 0.2 

def compute_costs(holding_units: float,
                  stockout_units: float,
                  ordered_units: float,
                  transferred_units: float,
                  p: CostParams):
    holding = holding_units * p.holding_cost
    stockout = stockout_units * p.stockout_cost
    order = ordered_units * p.order_cost
    transfer = transferred_units * p.transfer_cost
    total = holding + stockout + order + transfer
    return dict(
        holding=holding,
        stockout=stockout,
        order=order,
        transfer=transfer,
        total=total,
    )
