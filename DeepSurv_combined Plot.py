# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:31:41 2025

@author: Sarah Ogutu
"""

############################################################################
#combined plot
import matplotlib.pyplot as plt# side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(24, 9))

# Plot the survival predictions for the first model on the first axis
num_pts = 10
surv_pred.iloc[:, :num_pts].plot(ax=axes[0])
axes[0].set_ylabel('S(t | x)', fontsize = 30)
axes[0].set_title('(a)', fontsize = 30)
axes[0].set_xlabel('Time (Months)', fontsize = 30)
axes[0].set_ylim([0.75, 1])  # Set y-axis limit

# Adjust tick parameters for the first plot
axes[0].tick_params(axis='both', labelsize = 30)

# Increase the font size of the legend for the first plot
axes[0].legend(fontsize = 23)

# Plot the survival predictions for the second model on the second axis
surv_pred2.iloc[:, :num_pts].plot(ax=axes[1])
axes[1].set_ylabel('S(t | x)', fontsize = 30)
axes[1].set_title('(b)', fontsize = 30)
axes[1].set_xlabel('Time (Months)', fontsize = 30)
axes[1].set_ylim([0.75, 1])  # Set y-axis limit

# Adjust tick parameters for the second plot
axes[1].tick_params(axis='both', labelsize = 30 )

# Increase the font size of the legend for the first plot
axes[1].legend(fontsize = 23)

# Adjust the layout to prevent overlap and display the plot
plt.tight_layout()

# Save the combined plots as a high-resolution PDF
plt.savefig('D:/Downloads/Fred Hutch/DeepSurv.pdf', format='pdf', dpi=600)

plt.show()