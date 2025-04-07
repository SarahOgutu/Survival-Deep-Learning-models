# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 15:22:30 2025

@author: Sarah Ogutu
"""

#############################################################################

# combined plot
import matplotlib.pyplot as plt

# Create a figure with two subplots (side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))  # Adjust figsize as needed

# Plotting the first hazard graph in ax1
for i in range(hazard_df.shape[1]):
    ax1.plot(hazard_df.index, hazard_df.iloc[:, i], alpha=0.6)

ax1.set_title("(a)", fontsize = 24)
ax1.set_xlabel("Evaluation time (months)", fontsize = 30)
ax1.set_ylabel("Probability of risk", fontsize = 30)

# Plotting the second hazard graph in ax2 (replace with your own data for plot2)
for i in range(hazard_df2.shape[1]):  # Replace plot2_df with your actual DataFrame for plot2
    ax2.plot(hazard_df2.index, hazard_df2.iloc[:, i], alpha=0.6)  # Adjust as needed

ax2.set_title("(b)", fontsize = 24)  # Set your title for plot2
ax2.set_xlabel("Evaluation time (months)", fontsize = 30)
ax2.set_ylabel("Probability of risk", fontsize = 30)

# Increase font size for axis tick labels
ax1.tick_params(axis='both', labelsize=30) 
ax2.tick_params(axis='both', labelsize=30) 

# Set the same y-axis limits for both subplots
y_min, y_max = 0, 1  # Set appropriate limits based on your data
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Adjust layout
plt.tight_layout()

# Save the combined plots as a high-resolution PDF
plt.savefig('D:/Downloads/Fred Hutch/DeepHit.pdf', format='pdf', dpi=600)

# Show the combined plots
plt.show()