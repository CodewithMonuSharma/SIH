import os
import csv
import json
import time
from dataclasses import asdict
from typing import Dict, Any, List
from metrics.engine import PerformanceSummaryReport

class AutomaticLogger:
    """
    FR-25 Automatic Logging Subsystem.
    Writes per-frame telemetry data to CSV and run summary statistics to JSON/CSV
    automatically at the conclusion of every simulation run.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    def save_run_logs(self, frame_records: List[Dict[str, Any]], summary_report: PerformanceSummaryReport):
        """Writes both per-frame CSV and run summary JSON/CSV files."""
        # 1. Write per-frame CSV log
        csv_filename = os.path.join(self.log_dir, f"run_{self.timestamp_str}_frames.csv")
        if frame_records:
            fieldnames = list(frame_records[0].keys())
            with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(frame_records)

        # 2. Write per-run summary JSON report
        json_filename = os.path.join(self.log_dir, f"run_{self.timestamp_str}_summary.json")
        summary_dict = asdict(summary_report)
        with open(json_filename, mode="w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        print(f"[Logger] Saved per-frame log: {csv_filename}")
        print(f"[Logger] Saved run summary report: {json_filename}")
        return csv_filename, json_filename
