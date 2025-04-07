import json



import matplotlib.pyplot as plt

import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import numpy as np




dir_name = 'results/'

sitename_vec = ['hkn', 'hkz', 'bor', 'gem', 
                'god', 'vej', 'hr3', 'anh', 
                'kfl', 'bea','stb',
                'sea', 'mor', 'tri', 'ark',
                 'stn' ,'hs2','hs1']

res_all_sites = []

for site in sitename_vec:
    folder = dir_name+'res_'+site+'/'

    with open(folder+'data_res.json') as f:
        data_res = json.load(f)

    res_all = data_res['res_all']

    file_input = data_res['file_input']
    with open(file_input) as f:
        data_input = json.load(f)
           
    rel = data_input['rel']

    res_all_sites.append(res_all)
    


labels = ['Rule-based', 'Unlimited, perfect information', 'Perfect Information', 
              'Real Information', r'Pessimistic $\alpha$ = 1', 
              r'Pessimistic $\alpha$ = 2']

### Statistics of the results
ref_id = 3  # Index of the reference strategy for the calculation of relative differences
wdw_id = [3,4,5]
print('{}\tRevenue difference &\tReliability difference (compared to {})'.format("".ljust(32), labels[ref_id]))
print('{}\tMean\tMin\tMax\tMean\tMin\tMax'.format("Startegy".ljust(32)))
for i in wdw_id:
    mean_revenues = np.mean([(res_all[i]['revenues']/res_all[ref_id]['revenues']-1)*100  for res_all in res_all_sites])
    min_revenues = min([(res_all[i]['revenues']/res_all[ref_id]['revenues']-1)*100  for res_all in res_all_sites])
    max_revenues = max([(res_all[i]['revenues']/res_all[ref_id]['revenues']-1)*100  for res_all in res_all_sites])

    mean_reliability = np.mean([100*(res_all[i]['reliability'] - res_all[ref_id]['reliability']) for res_all in res_all_sites])
    min_reliability = min([100*(res_all[i]['reliability'] - res_all[ref_id]['reliability']) for res_all in res_all_sites]) 
    max_reliability = max([100*(res_all[i]['reliability'] - res_all[ref_id]['reliability']) for res_all in res_all_sites])

    print('{}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}pt\t{:.2f}pt\t{:.2f}pt'.format(labels[i].ljust(32), mean_revenues, min_revenues, max_revenues, mean_reliability, min_reliability, max_reliability))


ref_id = 2
print('Number of sites with reliability higher than: Target\t{}:'.format(labels[ref_id]))
for i in wdw_id:
    cnt_pi = 0
    for res_all in res_all_sites:
        if res_all[i]['reliability'] >= res_all[ref_id]['reliability']:
            cnt_pi+=1
    cnt_th = 0
    for res_all in res_all_sites:
        if res_all[i]['reliability'] >= rel:
            cnt_th+=1
    print('{}\t\t{:.0f}\t{:.0f}'.format(labels[i].ljust(32), cnt_th, cnt_pi))


# Plot the figure for all sites

col_cy = "#00A6D6"
col_db = "#0C2340"
col_dg = "#00B8C8"
col_bl = "#0076C2"
col_pp = "#6F1D77"
col_pk = "#EF60A3"
col_dr = "#A50034"
col_rd = "#E03C31"
col_or = "#EC6842"
col_ye = "#FFB81C"
col_lg = "#6CC24A"
col_dg = "#009B77"

SMALL_SIZE = 7
MEDIUM_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1

col_vec =  [col_db, col_dr, col_ye, col_lg, col_cy,  col_bl, col_db, col_db]

mrk_vec = ['s', '.', 'd', 'o', 'v', 's']


fig = plt.figure(figsize = (6.5, 2.5))

ref_id = 2 # Index of the reference strategy for the calculation of relative differences
wdw_id =  [3,4,5] # Indices of the strategies to be plotted

# ref_id = 1
# wdw_id =  [0,1, 2,3,4,5]

# Parameters for the dimensions of the plot
x_ax = 0.20
y_ax = 0.33
h_legend = 0.25
h_contour = 0.05
alpha_val =0.6
ax = fig.add_axes((x_ax,y_ax,1-x_ax-h_legend-h_contour,1-y_ax-h_contour))

cnt = 0
for res_all in res_all_sites:
    for i in wdw_id:

        if res_all[i]['reliability'] < res_all[1]['reliability']: 
            edge_color = col_vec[i]
        else:
            edge_color = 'black'

        # Calculate increase in revenues and reliability
        increase_rel = (res_all[i]['reliability'] - res_all[ref_id]['reliability'])*100 
        increase_rev = (res_all[i]['revenues']/res_all[ref_id]['revenues']-1)*100

        ax.plot(increase_rev, increase_rel, color = col_vec[i], linestyle = 'none', marker = mrk_vec[i], markersize=7, markerfacecolor = col_vec[i], markeredgecolor = edge_color, alpha = alpha_val)

    cnt+=1

ax_xlim = ax.get_xlim()
ax.plot(ax_xlim, [0, 0], 'k:')
ax.set_xlim(ax_xlim)

ax.set_xlabel(r'Relative difference in revenue [%]')
ax.set_ylabel(r'Difference in reliability [point]')

# create second Y-Axis.
dx_h = 0.04
ax2 = fig.add_axes((dx_h,y_ax,dx_h,1-y_ax-h_contour))
ax2.xaxis.set_visible(False) # hide the yaxis

new_tick_locations = np.array([.2, .5, .9])

for i in wdw_id[::-1]:
    vp = ax2.violinplot([100*(res_all[i]['reliability'] - res_all[ref_id]['reliability']) for res_all in res_all_sites], showmeans=False, showextrema=False, showmedians=False, widths = 20, bw_method=0.5, side='low')
    for pc in vp['bodies']:
        pc.set_facecolor(col_vec[i])
        pc.set_edgecolor(col_vec[i])
        pc.set_alpha(0.9)

ax_ylim = ax.get_ylim()
ax2.set_ylim(ax_ylim)

ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(right=True, labelleft= False,left=False)


# create second X-Axis.
dy_h = 0.08
ax3 = fig.add_axes((x_ax,dy_h,1-x_ax-h_contour-h_legend,dy_h))
ax3.yaxis.set_visible(False) # hide the yaxis

new_tick_locations = np.array([.2, .5, .9])

for i in wdw_id[::-1]:
    vp = ax3.violinplot([(res_all[i]['cost']/res_all[ref_id]['cost']-1)*100 for res_all in res_all_sites], showmeans=False, showextrema=False, showmedians=False, bw_method=0.5, side='low', vert=False)
    for pc in vp['bodies']:
        pc.set_facecolor(col_vec[i])
        pc.set_edgecolor(col_vec[i])
        pc.set_alpha(0.9)

ax_xlim = ax.get_xlim()
ax3.set_xlim(ax_xlim)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.tick_params(top=True, labelbottom= False,bottom=False)


# Create legend objects
point_lgd = []
text_lgd = []
for i in wdw_id:
    point_lgd.append(mlines.Line2D([], [], markersize = 7, marker = mrk_vec[i], markerfacecolor = col_vec[i], markeredgecolor = col_vec[i], alpha = 0.6, linestyle='None'))
    text_lgd.append(labels[i])

point_wc0b = mlines.Line2D([], [], markersize = 7, marker = mrk_vec[4], markerfacecolor = col_vec[4], markeredgecolor = 'black', alpha = 0.6, linestyle='None')
point_wc1b = mlines.Line2D([], [], markersize = 7, marker = mrk_vec[5], markerfacecolor = col_vec[5], markeredgecolor = 'black', alpha = 0.6, linestyle='None')

point_lgd.append( (point_wc0b, point_wc1b))
text_lgd.append('Reliability above target')

ax.legend(point_lgd, text_lgd, loc = 'upper left', bbox_to_anchor = (1,1), handler_map={tuple: HandlerTuple(ndivide=None, pad = 0.5)}, fontsize = 8)


plt.show(block = True)
