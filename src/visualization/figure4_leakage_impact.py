"""
Generate Data Leakage Impact figure for IEEE Transactions paper
Figure 4: Data Leakage Impact Comparison
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 5
categories = ['All 66 Features\n(with leakage)', '31 Clean Pre-Build\nFeatures']
accuracy = [0.9780, 0.8273]
roc_auc = [0.9956, 0.9138]
f1 = [0.9842, 0.8676]

# Set up positions
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))

# Create grouped bars
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, roc_auc, width, label='ROC-AUC', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c', alpha=0.8)

# Formatting
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Impact of Data Leakage Prevention on Random Forest Performance',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim([0.80, 1.01])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                '{:.4f}'.format(height), ha='center', va='bottom', fontsize=9)

# Add annotation for accuracy drop
ax.annotate('', xy=(1-width, 0.8273), xytext=(0-width, 0.9780),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(-0.5, 0.90, '15.07%\ndrop', fontsize=10, color='red',
        fontweight='bold', ha='center')

# Add text labels
ax.text(0, 0.82, 'Production\nVIABLE: NO', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
ax.text(1, 0.82, 'Production\nVIABLE: YES', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/lalitmishra/Workspace/Predicting-Software-System-Performance/figures/data_leakage_impact.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/lalitmishra/Workspace/Predicting-Software-System-Performance/figures/data_leakage_impact.png',
            format='png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved: data_leakage_impact.pdf and data_leakage_impact.png")
