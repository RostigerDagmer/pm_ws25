"""
Generate HTML and other format reports from evaluation metrics JSON.

This module separates presentation logic from evaluation logic,
allowing flexible report generation from pre-computed metrics.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional
import logging

# Path to data directory
DATA_DIR = Path("data")


class EvaluationReportGenerator:
    """Generate reports from evaluation metrics JSON."""

    def __init__(self, metrics_path: Optional[Path] = None, metrics_dict: Optional[Dict[str, Any]] = None,
                 baseline_comparison: Optional[Any] = None):
        """
        Initialize report generator from either JSON file or dict.

        Args:
            metrics_path: Path to metrics JSON file
            metrics_dict: Can be either:
                - Dict[str, EvaluationMetrics] (new format with per-dataset + 'overall')
                - Single EvaluationMetrics dict (legacy format)
            baseline_comparison: Optional pandas DataFrame with baseline comparison data
        """
        if metrics_path is None and metrics_dict is None:
            raise ValueError("Either metrics_path or metrics_dict must be provided")
        if metrics_path is not None and metrics_dict is not None:
            raise ValueError("Cannot provide both metrics_path and metrics_dict")

        if metrics_path:
            with open(metrics_path) as f:
                self.metrics = json.load(f)
        else:
            # Handle new format: Dict[str, EvaluationMetrics]
            if isinstance(metrics_dict, dict) and 'overall' in metrics_dict:
                # New format: Convert all EvaluationMetrics to dicts
                from models.evaluator import EvaluationMetrics
                self.all_metrics = {}
                for dataset_name, eval_metrics in metrics_dict.items():
                    if isinstance(eval_metrics, EvaluationMetrics):
                        self.all_metrics[dataset_name] = eval_metrics.to_dict()
                    else:
                        self.all_metrics[dataset_name] = eval_metrics
                # Set metrics to overall for backward compatibility
                self.metrics = self.all_metrics['overall']
            else:
                # Legacy format: single EvaluationMetrics
                self.metrics = metrics_dict
                self.all_metrics = {'overall': metrics_dict}

        # Store baseline comparison data
        self.baseline_comparison = baseline_comparison

    @classmethod
    def from_evaluation_metrics(cls, metrics):
        """Create report generator from EvaluationMetrics object or Dict[str, EvaluationMetrics]."""
        from models.evaluator import EvaluationMetrics
        if isinstance(metrics, dict):
            # Could be Dict[str, EvaluationMetrics] or single dict
            return cls(metrics_dict=metrics)
        elif isinstance(metrics, EvaluationMetrics):
            return cls(metrics_dict=metrics.to_dict())
        raise ValueError("metrics must be an EvaluationMetrics instance or dict")

    def to_html(self, output_path: Path) -> None:
        """Generate interactive HTML report."""
        html = self._generate_html()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logging.info(f"HTML report saved to: {output_path}")

    def _generate_html(self) -> str:
        """Generate complete HTML report."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>Heuristic Recommendation Evaluation Report</h1>
        <div class="nav-links">
            <a href="#per-dataset">Per-Dataset Breakdown</a>
            <a href="#per-heuristic">Per-Heuristic Analysis</a>
            <a href="#summary-table">Summary Table</a>
            <a href="#baseline-comparison">Baseline Comparison</a>
        </div>
    </nav>

    <div class="container">
        {self._generate_per_dataset_section()}
        {self._generate_per_heuristic_section()}
        {self._generate_summary_table_section()}
        {self._generate_baseline_comparison_section()}
    </div>

    <script>
        {self._get_javascript()}
    </script>
</body>
</html>"""

    def _get_css(self) -> str:
        """Get CSS styles for the HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .navbar {
            background: #2c3e50;
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .navbar h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        .nav-links a {
            color: white;
            text-decoration: none;
            margin-right: 1.5rem;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            transition: background-color 0.2s;
        }

        .nav-links a:hover {
            background-color: rgba(255,255,255,0.2);
        }

        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }

        .section {
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .section h3 {
            color: #34495e;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }

        .metric-card .label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }

        .metric-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .metric-card .unit {
            font-size: 1rem;
            color: #888;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
        }

        th {
            background: #34495e;
            color: white;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
        }

        td {
            padding: 0.8rem 1rem;
            border-bottom: 1px solid #e0e0e0;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .accuracy-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .accuracy-high {
            background-color: #d4edda;
            color: #155724;
        }

        .accuracy-medium {
            background-color: #fff3cd;
            color: #856404;
        }

        .accuracy-low {
            background-color: #f8d7da;
            color: #721c24;
        }

        .dataset-item {
            background: white;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .dataset-item.overall {
            border-color: #3498db;
            border-width: 3px;
            background-color: #f0f8ff;
        }

        .dataset-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            cursor: pointer;
            user-select: none;
        }

        .dataset-header h3 {
            margin: 0;
            color: #2c3e50;
        }

        .dataset-item.overall .dataset-header h3 {
            color: #2980b9;
        }

        .dataset-info {
            font-size: 1.1rem;
            font-weight: 500;
        }

        .expand-icon {
            font-size: 1.5rem;
            transition: transform 0.3s;
        }

        .expand-icon.expanded {
            transform: rotate(180deg);
        }

        .dataset-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }

        .dataset-content.expanded {
            max-height: 2000px;
        }

        .distribution-bar {
            background: #e9ecef;
            height: 30px;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
            border: 1px solid #dee2e6;
        }

        .distribution-segment {
            height: 100%;
            float: left;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .perf-ratio-good { color: #28a745; font-weight: bold; }
        .perf-ratio-ok { color: #ffc107; font-weight: bold; }
        .perf-ratio-bad { color: #dc3545; font-weight: bold; }

        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #dee2e6;
        }

        .tab-button {
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            color: #666;
            transition: all 0.2s;
        }

        .tab-button:hover {
            color: #2c3e50;
            background: #f8f9fa;
        }

        .tab-button.active {
            color: #3498db;
            border-bottom-color: #3498db;
        }

        .tab-content {
            display: none;
        }

        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .navbar h1 {
                font-size: 1.3rem;
            }

            .nav-links a {
                display: block;
                margin: 0.3rem 0;
            }
        }
        """

    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features."""
        return """
        // Tab switching
        function switchTab(evt, tabId) {
            // Hide all tab contents
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].style.display = 'none';
            }

            // Remove active class from all tab buttons
            const tabButtons = document.getElementsByClassName('tab-button');
            for (let i = 0; i < tabButtons.length; i++) {
                tabButtons[i].classList.remove('active');
            }

            // Show the selected tab content
            document.getElementById(tabId).style.display = 'block';

            // Add active class to the clicked button
            evt.currentTarget.classList.add('active');
        }

        // Toggle dataset details
        document.querySelectorAll('.dataset-header').forEach(header => {
            header.addEventListener('click', function() {
                const content = this.nextElementSibling;
                const icon = this.querySelector('.expand-icon');

                content.classList.toggle('expanded');
                icon.classList.toggle('expanded');
            });
        });

        // Sortable tables
        document.querySelectorAll('table.sortable th').forEach(header => {
            header.addEventListener('click', function() {
                const table = this.closest('table');
                const columnIndex = Array.from(this.parentElement.children).indexOf(this);
                sortTable(table, columnIndex);
            });
        });

        function sortTable(table, columnIndex) {
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            rows.sort((a, b) => {
                const aValue = a.cells[columnIndex].textContent.trim();
                const bValue = b.cells[columnIndex].textContent.trim();

                const aNum = parseFloat(aValue.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bValue.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return bNum - aNum; // Descending for numbers
                }
                return aValue.localeCompare(bValue);
            });

            rows.forEach(row => tbody.appendChild(row));
        }
        """

    def _generate_overall_section(self) -> str:
        """Generate the overall evaluation section."""
        metrics = self.metrics

        # Get overall accuracy values
        tolerance_metrics = metrics.get('tolerance_metrics', {})
        acc_0 = tolerance_metrics.get('0%', {}).get('overall_accuracy', 0) * 100
        acc_10 = tolerance_metrics.get('10%', {}).get('overall_accuracy', 0) * 100
        acc_20 = tolerance_metrics.get('20%', {}).get('overall_accuracy', 0) * 100

        # Performance metrics
        perf_ratio = metrics.get('performance_ratio_alignment_only', 1.0)
        time_savings = metrics.get('time_savings_vs_worst', 0) * 100

        return f"""
        <section id="overall" class="section">
            <h2>Overall Evaluation (All Test Datasets Combined)</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Accuracy @ 0% Tolerance</div>
                    <div class="value">{acc_0:.1f}<span class="unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Accuracy @ 10% Tolerance</div>
                    <div class="value">{acc_10:.1f}<span class="unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Accuracy @ 20% Tolerance</div>
                    <div class="value">{acc_20:.1f}<span class="unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Performance Ratio</div>
                    <div class="value {self._get_perf_ratio_class(perf_ratio)}">{perf_ratio:.3f}<span class="unit">x</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Time Savings vs Worst</div>
                    <div class="value">{time_savings:.1f}<span class="unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Samples</div>
                    <div class="value">{tolerance_metrics.get('0%', {}).get('total_samples', 0)}</div>
                </div>
            </div>

            <h3>Prediction Time</h3>
            <p>Mean Prediction Time: <strong>{metrics.get('mean_prediction_time', 0) * 1000:.2f}ms</strong></p>

            <h3>Top 5 Important Features</h3>
            {self._generate_feature_importance_table(metrics.get('feature_importance', {}))}
        </section>
        """

    def _generate_per_dataset_section(self) -> str:
        """Generate the per-dataset breakdown section."""
        datasets_html = []

        # First, add "Overall" as a special dataset item
        if 'overall' in self.all_metrics:
            datasets_html.append(self._generate_overall_dataset_item())

        # Then add individual datasets (all except 'overall')
        for dataset_name in sorted(self.all_metrics.keys()):
            if dataset_name != 'overall':
                dataset_metrics = self.all_metrics[dataset_name]
                datasets_html.append(self._generate_dataset_item_from_full_metrics(dataset_name, dataset_metrics))

        return f"""
        <section id="per-dataset" class="section">
            <h2>Per-Dataset Evaluation Breakdown</h2>
            {''.join(datasets_html)}
        </section>
        """

    def _generate_overall_dataset_item(self) -> str:
        """Generate HTML for the overall (aggregated) metrics as a dataset item."""
        metrics = self.metrics
        tolerance_metrics = metrics.get('tolerance_metrics', {})

        # Calculate overall label distribution from tolerance_metrics at 0%
        label_dist = {}
        if '0%' in tolerance_metrics:
            combo_metrics = tolerance_metrics['0%'].get('combination_metrics', {})
            for combo_str, combo_data in combo_metrics.items():
                # Parse combination string (e.g., "Dijkstra" or "A* + Dijkstra")
                heuristics = [h.strip() for h in combo_str.split('+')]
                # For simplicity, count the first heuristic in the combination
                if heuristics and heuristics[0]:
                    main_heuristic = heuristics[0]
                    label_dist[main_heuristic] = label_dist.get(main_heuristic, 0) + combo_data.get('support', 0)

        # Get prediction distribution
        pred_dist = tolerance_metrics.get('0%', {}).get('prediction_counts', {})

        # Build accuracy dict
        accuracy = {}
        for threshold_str, level_metrics in tolerance_metrics.items():
            accuracy[threshold_str] = level_metrics.get('overall_accuracy', 0)

        samples = tolerance_metrics.get('0%', {}).get('total_samples', 0)
        acc_0 = accuracy.get('0%', 0) * 100
        acc_10 = accuracy.get('10%', 0) * 100
        acc_20 = accuracy.get('20%', 0) * 100

        perf_ratio = metrics.get('performance_ratio_alignment_only', 1.0)
        time_savings = metrics.get('time_savings_vs_worst', 0) * 100

        mean_alignment = metrics.get('mean_alignment_time_only', 0)
        mean_optimal = metrics.get('mean_optimal_alignment_time', 0)

        # Find dominant heuristic
        dominant = max(label_dist.items(), key=lambda x: x[1]) if label_dist else ('N/A', 0)
        dominant_name, dominant_count = dominant
        dominant_pct = (dominant_count / samples * 100) if samples > 0 else 0

        # Get all features
        feature_importance = metrics.get('feature_importance', {})
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        features_html = '<ul style="margin: 0.5rem 0; padding-left: 1.5rem; column-count: 2; column-gap: 2rem;">'
        for fname, importance in sorted_features:
            features_html += f'<li><strong>{fname}</strong>: {importance:.4f}</li>'
        features_html += '</ul>'

        return f"""
        <div class="dataset-item overall">
            <div class="dataset-header">
                <div>
                    <h3>Overall (All Test Datasets Combined)</h3>
                    <span class="dataset-info" style="color: #666;">
                        {samples:,} samples |
                        Accuracy: {acc_0:.1f}% / {acc_10:.1f}% / {acc_20:.1f}% |
                        Perf Ratio: <span class="{self._get_perf_ratio_class(perf_ratio)}">{perf_ratio:.3f}x</span>
                    </span>
                </div>
                <span class="expand-icon">▼</span>
            </div>
            <div class="dataset-content">
                <div class="metrics-grid">
                    <div>
                        <strong>Mean Alignment Time:</strong><br>
                        {mean_alignment:.4f}s
                    </div>
                    <div>
                        <strong>Optimal Time:</strong><br>
                        {mean_optimal:.4f}s
                    </div>
                    <div>
                        <strong>Time Savings vs. Worst:</strong><br>
                        {time_savings:.1f}%
                    </div>
                </div>

                <h4>Label Distribution (Ground Truth)</h4>
                {self._generate_distribution_chart(label_dist, samples)}

                <h4>Prediction Distribution</h4>
                {self._generate_distribution_chart(pred_dist, samples)}

                {self._generate_legend()}

                <h4>Feature Importances</h4>
                {features_html}
            </div>
        </div>
        """

    def _generate_dataset_item_from_full_metrics(self, dataset_name: str, dataset_metrics: Dict[str, Any]) -> str:
        """Generate HTML for a single dataset item from full EvaluationMetrics."""
        # Assume dataset_name is already the display name
        display_name = dataset_name

        # Extract from tolerance_metrics
        tolerance_metrics = dataset_metrics.get('tolerance_metrics', {})
        samples = tolerance_metrics.get('0%', {}).get('total_samples', 0) if tolerance_metrics else 0

        # Get accuracies
        acc_0 = tolerance_metrics.get('0%', {}).get('overall_accuracy', 0) * 100 if '0%' in tolerance_metrics else 0
        acc_10 = tolerance_metrics.get('10%', {}).get('overall_accuracy', 0) * 100 if '10%' in tolerance_metrics else 0
        acc_20 = tolerance_metrics.get('20%', {}).get('overall_accuracy', 0) * 100 if '20%' in tolerance_metrics else 0

        perf_ratio = dataset_metrics.get('performance_ratio_alignment_only', 1.0)
        time_savings = dataset_metrics.get('time_savings_vs_worst', 0) * 100

        # Extract label and prediction distributions from tolerance_metrics
        # Label distribution: extract from combination_metrics
        label_dist = {}
        if '0%' in tolerance_metrics:
            combo_metrics = tolerance_metrics['0%'].get('combination_metrics', {})
            for combo_str, combo_data in combo_metrics.items():
                heuristics = [h.strip() for h in combo_str.split('+')]
                if heuristics and heuristics[0]:
                    main_heuristic = heuristics[0]
                    label_dist[main_heuristic] = label_dist.get(main_heuristic, 0) + combo_data.get('support', 0)

        pred_dist = tolerance_metrics.get('0%', {}).get('prediction_counts', {})

        # Find dominant heuristic
        dominant = max(label_dist.items(), key=lambda x: x[1]) if label_dist else ('N/A', 0)
        dominant_name, dominant_count = dominant
        dominant_pct = (dominant_count / samples * 100) if samples > 0 else 0

        return f"""
        <div class="dataset-item">
            <div class="dataset-header">
                <div>
                    <h3>{display_name}</h3>
                    <span class="dataset-info" style="color: #666;">
                        {samples} samples |
                        Accuracy: {acc_0:.1f}% / {acc_10:.1f}% / {acc_20:.1f}% |
                        Perf Ratio: <span class="{self._get_perf_ratio_class(perf_ratio)}">{perf_ratio:.3f}x</span>
                    </span>
                </div>
                <span class="expand-icon">▼</span>
            </div>
            <div class="dataset-content">
                <div class="metrics-grid">
                    <div>
                        <strong>Mean Alignment Time:</strong><br>
                        {dataset_metrics.get('mean_alignment_time_only', 0):.4f}s
                    </div>
                    <div>
                        <strong>Optimal Time:</strong><br>
                        {dataset_metrics.get('mean_optimal_alignment_time', 0):.4f}s
                    </div>
                    <div>
                        <strong>Time Savings vs. Worst:</strong><br>
                        {time_savings:.1f}%
                    </div>
                </div>

                <h4>Label Distribution (Ground Truth)</h4>
                {self._generate_distribution_chart(label_dist, samples)}

                <h4>Prediction Distribution</h4>
                {self._generate_distribution_chart(pred_dist, samples)}

                {self._generate_legend()}
            </div>
        </div>
        """

    def _generate_per_heuristic_section(self) -> str:
        """Generate per-heuristic performance analysis section."""
        tolerance_metrics = self.metrics.get('tolerance_metrics', {})

        if not tolerance_metrics:
            return ""

        # Create tabs for each tolerance level
        tolerance_tabs_html = []
        tolerance_content_html = []

        for idx, threshold_key in enumerate(['0%', '10%', '20%']):
            if threshold_key not in tolerance_metrics:
                continue

            level = tolerance_metrics[threshold_key]

            # Tab button
            active_class = "active" if idx == 0 else ""
            tolerance_tabs_html.append(f"""
                <button class="tab-button {active_class}" onclick="switchTab(event, 'tolerance-{threshold_key}')">{threshold_key} Tolerance</button>
            """)

            # Tab content
            display_style = "block" if idx == 0 else "none"

            if threshold_key == '0%':
                # For 0% tolerance: Show per-heuristic binary metrics
                per_heur = level.get('per_heuristic_metrics', {})

                if not per_heur:
                    continue

                # Sort heuristics by F1 score (descending)
                sorted_heuristics = sorted(
                    per_heur.items(),
                    key=lambda x: x[1].get('f1_score', 0),
                    reverse=True
                )

                # Create table rows
                heur_rows = []
                for heur_name, metrics in sorted_heuristics:
                    tp = metrics.get('true_positives', 0)
                    fp = metrics.get('false_positives', 0)
                    fn = metrics.get('false_negatives', 0)
                    precision = metrics.get('precision', 0) * 100
                    recall = metrics.get('recall', 0) * 100
                    accuracy = metrics.get('accuracy', 0) * 100
                    f1 = metrics.get('f1_score', 0) * 100
                    optimal_samples = metrics.get('total_optimal_samples', 0)

                    heur_rows.append(f"""
                    <tr>
                        <td><strong>{heur_name}</strong></td>
                        <td style="text-align: right;">{optimal_samples}</td>
                        <td style="text-align: right;">{tp}</td>
                        <td style="text-align: right;">{fp}</td>
                        <td style="text-align: right;">{fn}</td>
                        <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(precision)}">{precision:.1f}%</span></td>
                        <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(recall)}">{recall:.1f}%</span></td>
                        <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(f1)}">{f1:.1f}%</span></td>
                        <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(accuracy)}">{accuracy:.1f}%</span></td>
                    </tr>
                    """)

                tolerance_content_html.append(f"""
                <div id="tolerance-{threshold_key}" class="tab-content" style="display: {display_style};">
                    <p style="margin-bottom: 1rem; color: #666;">
                        Binary classification metrics (One-vs-Rest) for each heuristic at {threshold_key} tolerance level.
                        At 0% tolerance, the ground truth is unambiguous (single best heuristic), making binary metrics fully interpretable.
                    </p>
                    <table class="sortable">
                        <thead>
                            <tr>
                                <th>Heuristic</th>
                                <th style="text-align: right;" title="Number of samples where this heuristic is optimal">Optimal Samples</th>
                                <th style="text-align: right;" title="True Positives: Predicted H and H is optimal">TP</th>
                                <th style="text-align: right;" title="False Positives: Predicted H but H is not optimal">FP</th>
                                <th style="text-align: right;" title="False Negatives: Did not predict H but H is optimal">FN</th>
                                <th style="text-align: right;" title="TP / (TP + FP): When we predict H, how often is it correct?">Precision</th>
                                <th style="text-align: right;" title="TP / (TP + FN): When H is optimal, how often do we predict it?">Recall</th>
                                <th style="text-align: right;" title="Harmonic mean of Precision and Recall">F1 Score</th>
                                <th style="text-align: right;" title="(TP + TN) / Total: Overall correctness">Accuracy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(heur_rows)}
                        </tbody>
                    </table>
                </div>
                """)
            else:
                # For 10%/20% tolerance: Show per-combination metrics
                combo_metrics = level.get('combination_metrics', {})

                if not combo_metrics:
                    continue

                # Sort combinations by support (descending)
                sorted_combos = sorted(
                    combo_metrics.items(),
                    key=lambda x: x[1].get('support', 0),
                    reverse=True
                )

                # Create table rows
                combo_rows = []
                for combo_name, metrics in sorted_combos:
                    support = metrics.get('support', 0)
                    correct = metrics.get('correct_predictions', 0)
                    recall = metrics.get('recall', 0) * 100

                    combo_rows.append(f"""
                    <tr>
                        <td><strong>{combo_name}</strong></td>
                        <td style="text-align: right;">{support}</td>
                        <td style="text-align: right;">{correct}</td>
                        <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(recall)}">{recall:.1f}%</span></td>
                    </tr>
                    """)

                tolerance_content_html.append(f"""
                <div id="tolerance-{threshold_key}" class="tab-content" style="display: {display_style};">
                    <p style="margin-bottom: 1rem; color: #666;">
                        Per-combination ground truth metrics at {threshold_key} tolerance level.
                        At non-zero tolerance, multiple heuristics can be near-optimal, forming combinations.
                        Shows how well the classifier performs when specific combinations of heuristics are optimal.
                    </p>
                    <table class="sortable">
                        <thead>
                            <tr>
                                <th>Near-Optimal Combination</th>
                                <th style="text-align: right;" title="Number of samples with this near-optimal combination">Support</th>
                                <th style="text-align: right;" title="How often we predicted a heuristic from this combination">Correct Predictions</th>
                                <th style="text-align: right;" title="Correct Predictions / Support: When this combination is optimal, how often do we predict any member of it? (Combination Recall)">Recall</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(combo_rows)}
                        </tbody>
                    </table>
                </div>
                """)

        return f"""
        <section id="per-heuristic" class="section">
            <h2>Per-Heuristic & Per-Combination Analysis</h2>
            <p style="margin-bottom: 1rem;">
                <strong>0% Tolerance:</strong> Binary classification metrics per heuristic (unambiguous ground truth).<br>
                <strong>10%/20% Tolerance:</strong> Per-combination metrics showing performance when multiple heuristics are near-optimal.
            </p>

            <div class="tabs">
                {''.join(tolerance_tabs_html)}
            </div>

            {''.join(tolerance_content_html)}
        </section>
        """

    def _generate_summary_table_section(self) -> str:
        """Generate the summary table comparing all datasets."""
        rows = []

        # First row: Overall
        if 'overall' in self.all_metrics:
            overall_metrics = self.all_metrics['overall']
            tolerance_metrics = overall_metrics.get('tolerance_metrics', {})

            overall_samples = tolerance_metrics.get('0%', {}).get('total_samples', 0)
            overall_acc_0 = tolerance_metrics.get('0%', {}).get('overall_accuracy', 0) * 100
            overall_acc_10 = tolerance_metrics.get('10%', {}).get('overall_accuracy', 0) * 100
            overall_acc_20 = tolerance_metrics.get('20%', {}).get('overall_accuracy', 0) * 100
            overall_perf_ratio = overall_metrics.get('performance_ratio_alignment_only', 1.0)

            # Calculate overall label distribution from combination metrics
            overall_label_dist = {}
            if '0%' in tolerance_metrics:
                combo_metrics = tolerance_metrics['0%'].get('combination_metrics', {})
                for combo_str, combo_data in combo_metrics.items():
                    heuristics = [h.strip() for h in combo_str.split('+')]
                    if heuristics and heuristics[0]:
                        main_heuristic = heuristics[0]
                        overall_label_dist[main_heuristic] = overall_label_dist.get(main_heuristic, 0) + combo_data.get('support', 0)

            # Get overall prediction distribution
            overall_pred_dist = tolerance_metrics.get('0%', {}).get('prediction_counts', {})

            overall_label_chart = self._generate_distribution_chart(overall_label_dist, overall_samples)
            overall_pred_chart = self._generate_distribution_chart(overall_pred_dist, overall_samples)

            rows.append(f"""
            <tr style="background-color: #f0f8ff; font-weight: 600;">
                <td><strong>Overall</strong></td>
                <td style="text-align: right;">{overall_samples:,}</td>
                <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(overall_acc_0)}">{overall_acc_0:.1f}%</span></td>
                <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(overall_acc_10)}">{overall_acc_10:.1f}%</span></td>
                <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(overall_acc_20)}">{overall_acc_20:.1f}%</span></td>
                <td style="text-align: right;"><span class="{self._get_perf_ratio_class(overall_perf_ratio)}">{overall_perf_ratio:.3f}</span></td>
                <td>
                    <div style="margin-bottom: 0.5rem;">
                        <small style="color: #666; font-weight: normal;">Ground Truth:</small>
                        {overall_label_chart}
                    </div>
                    <div>
                        <small style="color: #666; font-weight: normal;">Predicted:</small>
                        {overall_pred_chart}
                    </div>
                </td>
            </tr>
            """)

        # Individual datasets (all except 'overall')
        for dataset_name in sorted(self.all_metrics.keys()):
            if dataset_name == 'overall':
                continue

            dataset_metrics = self.all_metrics[dataset_name]
            # Assume dataset_name is already display name
            display_name = dataset_name

            tolerance_metrics = dataset_metrics.get('tolerance_metrics', {})
            samples = tolerance_metrics.get('0%', {}).get('total_samples', 0)
            acc_0 = tolerance_metrics.get('0%', {}).get('overall_accuracy', 0) * 100
            acc_10 = tolerance_metrics.get('10%', {}).get('overall_accuracy', 0) * 100
            acc_20 = tolerance_metrics.get('20%', {}).get('overall_accuracy', 0) * 100
            perf_ratio = dataset_metrics.get('performance_ratio_alignment_only', 1.0)

            # Extract label distribution
            label_dist = {}
            if '0%' in tolerance_metrics:
                combo_metrics = tolerance_metrics['0%'].get('combination_metrics', {})
                for combo_str, combo_data in combo_metrics.items():
                    heuristics = [h.strip() for h in combo_str.split('+')]
                    if heuristics and heuristics[0]:
                        main_heuristic = heuristics[0]
                        label_dist[main_heuristic] = label_dist.get(main_heuristic, 0) + combo_data.get('support', 0)

            pred_dist = tolerance_metrics.get('0%', {}).get('prediction_counts', {})
            label_chart = self._generate_distribution_chart(label_dist, samples)
            pred_chart = self._generate_distribution_chart(pred_dist, samples)

            rows.append(f"""
            <tr>
                <td>{display_name}</td>
                <td style="text-align: right;">{samples:,}</td>
                <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(acc_0)}">{acc_0:.1f}%</span></td>
                <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(acc_10)}">{acc_10:.1f}%</span></td>
                <td style="text-align: right;"><span class="{self._get_accuracy_badge_class(acc_20)}">{acc_20:.1f}%</span></td>
                <td style="text-align: right;"><span class="{self._get_perf_ratio_class(perf_ratio)}">{perf_ratio:.3f}</span></td>
                <td>
                    <div style="margin-bottom: 0.5rem;">
                        <small style="color: #666;">Ground Truth:</small>
                        {label_chart}
                    </div>
                    <div>
                        <small style="color: #666;">Predicted:</small>
                        {pred_chart}
                    </div>
                </td>
            </tr>
            """)

        return f"""
        <section id="summary-table" class="section">
            <h2>Summary Table: Per-Dataset Performance</h2>
            {self._generate_legend()}
            <table class="sortable">
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th style="text-align: right;">Samples</th>
                        <th style="text-align: right;">Acc @ 0%</th>
                        <th style="text-align: right;">Acc @ 10%</th>
                        <th style="text-align: right;">Acc @ 20%</th>
                        <th style="text-align: right;">Perf Ratio</th>
                        <th style="min-width: 300px;">Label Distribution</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </section>
        """

    def _generate_feature_importance_table(self, feature_importance: Dict[str, float]) -> str:
        """Generate HTML table for top feature importances."""
        if not feature_importance:
            return "<p><em>No feature importance data available.</em></p>"

        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        rows = []
        for feature, importance in sorted_features:
            bar_width = importance * 100  # Assuming importance is 0-1
            rows.append(f"""
            <tr>
                <td>{feature}</td>
                <td style="text-align: right;">{importance:.4f}</td>
                <td>
                    <div style="background: #e9ecef; border-radius: 4px; height: 20px; width: 200px; border: 1px solid #dee2e6;">
                        <div style="background: #3498db;
                                    height: 100%; width: {bar_width}%; border-radius: 4px;"></div>
                    </div>
                </td>
            </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th style="text-align: right;">Importance</th>
                    <th>Visualization</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _generate_distribution_chart(self, distribution: Dict[str, int], total: int) -> str:
        """Generate a horizontal bar chart for heuristic distribution without labels."""
        if not distribution or total == 0:
            return "<p><em>No data</em></p>"

        sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        # Simple color palette for different heuristics
        colors = {
            'Dijkstra': '#3498db',
            'A*-ILP': '#9b59b6',
            'A*': '#e74c3c',
            'RemainingActivities': '#1abc9c',
            'RequiredModelMove': '#f39c12',
        }

        segments = []
        for heuristic, count in sorted_dist:
            percentage = (count / total * 100) if total > 0 else 0
            color = colors.get(heuristic, '#999')
            # Tooltip with heuristic name and percentage
            title = f"{heuristic}: {count} ({percentage:.1f}%)"
            segments.append(
                f'<div class="distribution-segment" style="width: {percentage}%; background: {color};" title="{title}"></div>'
            )

        return f'<div class="distribution-bar">{"".join(segments)}</div>'

    def _generate_legend(self) -> str:
        """Generate legend for heuristic colors."""
        colors = {
            'Dijkstra': '#3498db',
            'RemainingActivities': '#1abc9c',
            'A*-ILP': '#9b59b6',
            'A*': '#e74c3c',
            'RequiredModelMove': '#f39c12',
        }

        legend_items = []
        for heuristic, color in colors.items():
            legend_items.append(
                f'<span style="display: inline-flex; align-items: center; margin-right: 1rem;">'
                f'<span style="display: inline-block; width: 20px; height: 20px; background: {color}; '
                f'border-radius: 3px; margin-right: 0.3rem; border: 1px solid #dee2e6;"></span>'
                f'<span style="font-size: 0.9rem;">{heuristic}</span>'
                f'</span>'
            )

        return f'<div style="margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6;">{"".join(legend_items)}</div>'

    def _get_accuracy_badge_class(self, accuracy: float) -> str:
        """Get CSS class for accuracy badge based on value."""
        if accuracy >= 85:
            return "accuracy-badge accuracy-high"
        elif accuracy >= 70:
            return "accuracy-badge accuracy-medium"
        else:
            return "accuracy-badge accuracy-low"

    def _get_perf_ratio_class(self, ratio: float) -> str:
        """Get CSS class for performance ratio."""
        if ratio <= 1.1:
            return "perf-ratio-good"
        elif ratio <= 1.3:
            return "perf-ratio-ok"
        else:
            return "perf-ratio-bad"

    def _get_perf_overhead_class(self, overhead_pct: float) -> str:
        """Get CSS class for performance overhead percentage."""
        if overhead_pct <= 10.0:
            return "perf-ratio-good"
        elif overhead_pct <= 30.0:
            return "perf-ratio-ok"
        else:
            return "perf-ratio-bad"

    def _generate_baseline_comparison_section(self) -> str:
        """Generate baseline comparison section with tabs."""
        if self.baseline_comparison is None:
            return """
            <section id="baseline-comparison" class="section">
                <h2>Baseline Comparison</h2>
                <p><em>No baseline comparison data available.</em></p>
            </section>
            """

        # Convert DataFrame to dict for easier processing
        import pandas as pd
        if isinstance(self.baseline_comparison, pd.DataFrame):
            df = self.baseline_comparison
        else:
            return """
            <section id="baseline-comparison" class="section">
                <h2>Baseline Comparison</h2>
                <p><em>Invalid baseline comparison data format.</em></p>
            </section>
            """

        # Extract model names
        models = df['model'].tolist()

        # Tab buttons HTML
        tabs_html = []
        tabs_html.append('<button class="tab-button active" onclick="switchTab(event, \'baseline-overall\')">Overall Performance</button>')

        # Extract tolerance levels from column names
        tolerance_levels = []
        for col in df.columns:
            if col.startswith('accuracy_'):
                tolerance = col.replace('accuracy_', '')
                if tolerance not in tolerance_levels:
                    tolerance_levels.append(tolerance)

        for tolerance in sorted(tolerance_levels):
            tabs_html.append(f'<button class="tab-button" onclick="switchTab(event, \'baseline-{tolerance}\')">Tolerance {tolerance}</button>')

        # Generate tab contents
        tab_contents = []

        # Overall Performance Tab
        overall_tab = self._generate_overall_performance_tab(df, models)
        tab_contents.append(f'<div id="baseline-overall" class="tab-content" style="display: block;">{overall_tab}</div>')

        # Per-tolerance tabs
        for idx, tolerance in enumerate(sorted(tolerance_levels)):
            tolerance_tab = self._generate_tolerance_tab(df, models, tolerance)
            tab_contents.append(f'<div id="baseline-{tolerance}" class="tab-content" style="display: none;">{tolerance_tab}</div>')

        return f"""
        <section id="baseline-comparison" class="section">
            <h2>Baseline Comparison</h2>
            <p>Comparison of the main classifier against baseline models across different metrics.</p>

            <div class="tabs">
                {''.join(tabs_html)}
            </div>

            {''.join(tab_contents)}
        </section>
        """

    def _generate_overall_performance_tab(self, df, models) -> str:
        """Generate the overall performance comparison tab."""
        rows = []

        for _, row in df.iterrows():
            model_name = row['model']
            perf_ratio_alignment = row.get('performance_ratio_alignment_only', 'N/A')
            perf_ratio_with_pred = row.get('performance_ratio_with_prediction', 'N/A')
            mean_alignment = row.get('mean_alignment_time_only', 'N/A')
            mean_alignment_with_pred = row.get('mean_alignment_time_with_prediction', 'N/A')
            mean_pred_time = row.get('mean_prediction_time', 'N/A')
            mean_fe_time = row.get('mean_feature_extraction_time', 'N/A')
            mean_clf_time = row.get('mean_classification_time', 'N/A')

            # Calculate percentage overhead from performance ratio
            # If ratio is 1.2, we are 20% slower than optimal
            perf_overhead_alignment = ((perf_ratio_alignment - 1.0) * 100) if isinstance(perf_ratio_alignment, (int, float)) else None
            perf_overhead_with_pred = ((perf_ratio_with_pred - 1.0) * 100) if isinstance(perf_ratio_with_pred, (int, float)) else None

            # Format values - all times in milliseconds
            if perf_overhead_alignment is not None:
                if perf_overhead_alignment >= 0:
                    perf_overhead_alignment_str = f"+{perf_overhead_alignment:.1f}%"
                else:
                    perf_overhead_alignment_str = f"{perf_overhead_alignment:.1f}%"
            else:
                perf_overhead_alignment_str = "N/A"

            if perf_overhead_with_pred is not None:
                if perf_overhead_with_pred >= 0:
                    perf_overhead_with_pred_str = f"+{perf_overhead_with_pred:.1f}%"
                else:
                    perf_overhead_with_pred_str = f"{perf_overhead_with_pred:.1f}%"
            else:
                perf_overhead_with_pred_str = "N/A"

            mean_fe_str = f"{mean_fe_time * 1000:.2f}" if isinstance(mean_fe_time, (int, float)) else "N/A"
            mean_clf_str = f"{mean_clf_time * 1000:.2f}" if isinstance(mean_clf_time, (int, float)) else "N/A"
            mean_alignment_str = f"{mean_alignment * 1000:.2f}" if isinstance(mean_alignment, (int, float)) else "N/A"
            mean_total_str = f"{mean_alignment_with_pred * 1000:.2f}" if isinstance(mean_alignment_with_pred, (int, float)) else "N/A"

            # Get class for performance overhead (use same thresholds: 10%, 30%)
            perf_class_alignment = self._get_perf_overhead_class(perf_overhead_alignment) if perf_overhead_alignment is not None else ""
            perf_class_with_pred = self._get_perf_overhead_class(perf_overhead_with_pred) if perf_overhead_with_pred is not None else ""

            rows.append(f"""
            <tr>
                <td><strong>{model_name}</strong></td>
                <td style="text-align: right;"><span class="{perf_class_alignment}">{perf_overhead_alignment_str}</span></td>
                <td style="text-align: right;"><span class="{perf_class_with_pred}">{perf_overhead_with_pred_str}</span></td>
                <td style="text-align: right;">{mean_fe_str}</td>
                <td style="text-align: right;">{mean_clf_str}</td>
                <td style="text-align: right;">{mean_alignment_str}</td>
                <td style="text-align: right;">{mean_total_str}</td>
            </tr>
            """)

        return f"""
        <p style="margin-bottom: 1rem; color: #666;">
            Performance comparison across all models. All times are in <strong>milliseconds</strong>.
            <br><strong>Overhead vs Optimal:</strong> How much slower (%) compared to always choosing the optimal heuristic (0% is perfect).
            <br><strong>Total Time:</strong> Feature Extraction + Classification + Alignment Time (complete end-to-end time).
        </p>
        <table class="sortable">
            <thead>
                <tr>
                    <th>Model</th>
                    <th style="text-align: right;" title="How much slower (%) than optimal - alignment time only">Overhead vs Optimal<br>(Alignment Only)</th>
                    <th style="text-align: right;" title="How much slower (%) than optimal - including prediction overhead">Overhead vs Optimal<br>(With Prediction)</th>
                    <th style="text-align: right;" title="Mean time for feature extraction">Feature Extraction<br>(ms)</th>
                    <th style="text-align: right;" title="Mean time for classification">Classification<br>(ms)</th>
                    <th style="text-align: right;" title="Mean alignment execution time">Alignment Time<br>(ms)</th>
                    <th style="text-align: right;" title="Feature Extraction + Classification + Alignment (end-to-end)">Total Time<br>(ms)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _generate_tolerance_tab(self, df, models, tolerance: str) -> str:
        """Generate a tolerance-specific comparison tab."""
        rows = []

        for _, row in df.iterrows():
            model_name = row['model']
            accuracy_col = f'accuracy_{tolerance}'
            macro_accuracy_col = f'macro_accuracy_{tolerance}'

            accuracy = row.get(accuracy_col, 'N/A')
            macro_accuracy = row.get(macro_accuracy_col, 'N/A')

            # Format values
            accuracy_str = f"{accuracy * 100:.1f}%" if isinstance(accuracy, (int, float)) else accuracy
            macro_accuracy_str = f"{macro_accuracy * 100:.1f}%" if isinstance(macro_accuracy, (int, float)) else macro_accuracy

            # Get badge classes
            accuracy_class = self._get_accuracy_badge_class(accuracy * 100) if isinstance(accuracy, (int, float)) else ""
            macro_class = self._get_accuracy_badge_class(macro_accuracy * 100) if isinstance(macro_accuracy, (int, float)) else ""

            rows.append(f"""
            <tr>
                <td><strong>{model_name}</strong></td>
                <td style="text-align: right;"><span class="{accuracy_class}">{accuracy_str}</span></td>
                <td style="text-align: right;"><span class="{macro_class}">{macro_accuracy_str}</span></td>
            </tr>
            """)

        return f"""
        <p style="margin-bottom: 1rem; color: #666;">
            Accuracy metrics at <strong>{tolerance}</strong> tolerance level.
            <strong>Micro (Overall) Accuracy:</strong> Weighted by sample frequency (common combinations dominate).
            <strong>Macro Accuracy:</strong> Unweighted average across all combinations (treats rare combinations equally).
        </p>
        <table class="sortable">
            <thead>
                <tr>
                    <th>Model</th>
                    <th style="text-align: right;" title="Overall accuracy: correct predictions / total samples">Micro Accuracy (Overall)</th>
                    <th style="text-align: right;" title="Average recall across all combinations (unweighted)">Macro Accuracy</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
