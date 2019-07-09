'''
plot a legend and other elements of the "trajectories" figure
'''
import os
import numpy as np
import matplotlib.pyplot as plt

import pose_estimation.directories as dirs
import pose_estimation.gramian.trajectories as gt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# legend box plot
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
els = [0,2,4] # elements of gt.tex_names to use
for i in els:
    lbl = gt.tex_names[i]
    ax.scatter(0, 0, c=[gt.clrs[i]], label=lbl, s=400) 

# plot invisible point to shift actual points to the left 
# (and out of the way of the plot)
ax.scatter(1, 0, c=[[0,0,0,0]], s=1)
ax.axis('off')

# make legend and save
txt_clr = [1,1,1]
lgnd = ax.legend(loc='upper right', framealpha=1.0, frameon=True,
                 edgecolor=txt_clr, facecolor='none', handletextpad=.1,
                 labelspacing=.3, borderpad=.2, prop={'size':20})
plt.setp(lgnd.get_texts(), color=txt_clr)
legend_file = os.path.join(dirs.trajectories_dir, 'legend.png')
fig.savefig(legend_file, transparent=True)

# plot the words 'min' and 'max', which will later be cropped and added to the
# bottom of the figure
fig_1 = plt.figure(dpi=300)
ax_1 = fig_1.add_subplot(111)
ax_1.axis('off')
ax_1.text(0, 0, 'min', fontdict={'size':200, 'color':txt_clr})
ax_1.text(0, .6, 'max', fontdict={'size':200, 'color':txt_clr})
min_max_text_file = os.path.join(dirs.trajectories_dir, 'min_max_text.png')
fig_1.savefig(min_max_text_file, transparent=True)
plt.show()
