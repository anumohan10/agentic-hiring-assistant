import json
import os
from datetime import datetime

def save_simulation_log(log_data, policy_name, out_dir="logs"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{policy_name}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    return out_path
