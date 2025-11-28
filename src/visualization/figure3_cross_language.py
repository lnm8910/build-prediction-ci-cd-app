"""
Generate Cross-Language Performance figure for IEEE Transactions paper
Figure 3: Cross-Language Performance Comparison
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 4
languages = ['Java', 'Ruby', 'Python', 'JavaScript']
accuracy = [0.8421, 0.8214, 0.8154, 0.8083]
precision = [0.8245, 0.8102, 0.7998, 0.7923]
recall = [0.9412, 0.9301, 0.9385, 0.9298]
f1 = [0.8790, 0.8661, 0.8639, 0.8556]

# Set up positions
x = np.arange(len(languages))
width = 0.2  # Width of bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#2ca02c', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#d62728', alpha=0.8)

# Formatting
ax.set_xlabel('Programming Language', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Cross-Language Build Prediction Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(languages, fontsize=11)
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim([0.75, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add horizontal line for overall accuracy
ax.axhline(y=0.8274, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.text(3.5, 0.83, 'Overall: 0.8274', fontsize=9, ha='right', style='italic')

plt.tight_layout()
plt.savefig('/Users/lalitmishra/Workspace/Predicting-Software-System-Performance/figures/cross_language_performance.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/lalitmishra/Workspace/Predicting-Software-System-Performance/figures/cross_language_performance.png',
            format='png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 saved: cross_language_performance.pdf and cross_language_performance.png")
