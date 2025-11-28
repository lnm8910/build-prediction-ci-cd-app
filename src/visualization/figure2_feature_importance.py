"""
Generate Feature Importance figure for IEEE Transactions paper
Figure 2: Feature Importance Bar Chart
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data from Table 3 (Top 15 features)
features = [
    'gh_project_maturity_days',
    'git_repository_age_days',
    'gh_commits_count',
    'gh_total_commits',
    'gh_sloc',
    'gh_contributors_count',
    'tr_build_number',
    'gh_test_density',
    'gh_tests_count',
    'tr_builds_last_30_days',
    'gh_src_complexity_avg',
    'gh_test_coverage_previous',
    'tr_prev_build_success',
    'gh_nesting_depth_avg',
    'git_num_files_modified'
]

importance = [0.0949, 0.0946, 0.0902, 0.0863, 0.0766, 0.0654,
              0.0612, 0.0587, 0.0543, 0.0498, 0.0487, 0.0456,
              0.0423, 0.0401, 0.0389]

# Categories for color coding
categories = [
    'Project Context', 'Project Context', 'Project Context', 'Project Context',
    'Code Metrics', 'Project Context', 'Build History', 'Test Structure',
    'Test Structure', 'Build History', 'Code Metrics', 'Test Structure',
    'Build History', 'Code Metrics', 'Commit Context'
]

# Color mapping
color_map = {
    'Project Context': '#1f77b4',    # Blue
    'Code Metrics': '#ff7f0e',       # Orange
    'Test Structure': '#2ca02c',     # Green
    'Build History': '#d62728',      # Red
    'Commit Context': '#9467bd'      # Purple
}
colors = [color_map[cat] for cat in categories]

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Important Pre-Build Features for Build Prediction',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()  # Highest importance at top
ax.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, importance)):
    ax.text(val + 0.002, i, '{:.4f}'.format(val), va='center', fontsize=8)

# Legend
legend_elements = [
    Patch(facecolor='#1f77b4', label='Project Context (49.8%)'),
    Patch(facecolor='#2ca02c', label='Test Structure (11.3%)'),
    Patch(facecolor='#d62728', label='Build History (10.1%)'),
    Patch(facecolor='#ff7f0e', label='Code Metrics (7.7%)'),
    Patch(facecolor='#9467bd', label='Commit Context')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/lalitmishra/Workspace/Predicting-Software-System-Performance/figures/feature_importance.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/lalitmishra/Workspace/Predicting-Software-System-Performance/figures/feature_importance.png',
            format='png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 saved: feature_importance.pdf and feature_importance.png")
