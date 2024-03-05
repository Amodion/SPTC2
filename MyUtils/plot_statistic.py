import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_stats(num_epochs, bbox_stats, kps_stats, loss_bb, loss_kp, loss, num=None, show=False, log_path=None):

    
    sns.set_theme()
    sns.set_style("darkgrid")
    
#    stats = pd.DataFrame({
#        'Epoch': range(num_epochs),
#        'Bbox_stat': bbox_stats,
#        'Kps_stat': kps_stats})
    
    if 1 <= num_epochs <= 100:
        step = 1
    else:
        step = 10
    
    stats_bb = pd.DataFrame(data=bbox_stats[::step], columns=['IoU=0.50:0.95 | area = all', 'IoU=0.50         | area = all', 'IoU=0.75         | area = all', 'IoU=0.50:0.95 | area = small', 'IoU=0.50:0.95 | area = medium', 'IoU=0.50:0.95 | area = large'])
    stats_bb['Epoch'] = range(num_epochs)[::step]
    stats_bb_melt = pd.melt(stats_bb, id_vars=['Epoch'])
    
    stats_kp = pd.DataFrame(data=kps_stats[::step], columns=['IoU=0.50:0.95 | area = all', 'IoU=0.50         | area = all', 'IoU=0.75         | area = all', 'IoU=0.50:0.95 | area = small', 'IoU=0.50:0.95 | area = medium', 'IoU=0.50:0.95 | area = large'])
    stats_kp['Epoch'] = range(num_epochs)[::step]
    stats_kp_melt = pd.melt(stats_kp, id_vars=['Epoch'])
    
#    print(bbox_stats)
#    print(kps_stats)
#    print([loss_bb, loss_kp])
    losses = pd.DataFrame()
    losses['Epoch'] = range(num_epochs)[::step]
    losses['Bounding Box loss'] = loss_bb[::step]
    losses['Keypoints loss'] = loss_kp[::step]
    losses['Loss general'] = loss[::step]
    losses_melt = pd.melt(losses, id_vars=['Epoch'])

#    stats_melt = pd.melt(stats, id_vars=['Epoch'])

    figure, axes = plt.subplots(3, 1, figsize=(40, 40))
#    print(axes)

    figure.suptitle('Metrics', fontsize=22)
    first = sns.pointplot(data=stats_bb_melt, x='Epoch', y='value', hue='variable', ax=axes[0])#, style='variable', markers=True, dashes=False
#    first.set(xticks=np.arange(0,num_epochs,1))
#NPoints_hist.bar_label(NPoints_hist.containers[0], fontsize=10)
    axes[0].xaxis.grid(True)
    axes[0].set_ylim(ymin=-0.01)
    axes[0].legend(loc='upper left', fontsize='x-large')
    axes[0].set_title('Average Precision of Bounding boxes', fontsize=18)
    axes[0].set_xlabel('Epoch\n\n\n', fontsize=18)
    axes[0].set_ylabel('Metrics values', fontsize=18)

    second = sns.pointplot(data=stats_kp_melt, x='Epoch', y='value', hue='variable', ax=axes[1])
#    second.set(xticks=np.arange(0,num_epochs,1))
    axes[1].xaxis.grid(True)
    axes[1].set_ylim(ymin=-0.01)
    axes[1].legend(loc='upper left', fontsize='x-large')
    axes[1].set_title('Average Precision of Keypoints', fontsize=18)
    axes[1].set_xlabel('Epoch', fontsize=18)
    axes[1].set_ylabel('Metric values', fontsize=18)
    
    third = sns.pointplot(data=losses_melt, x='Epoch', y='value', hue='variable', ax=axes[2])
#    second.set(xticks=np.arange(0,num_epochs,1))
    axes[2].xaxis.grid(True)
#    axes[2].set_ylim(ymin=-0.01)
    axes[2].legend(loc='upper left', fontsize='x-large')
    axes[2].set_title('Losses', fontsize=18)
    axes[2].set_xlabel('Epoch', fontsize=18)
    axes[2].set_ylabel('Loss values', fontsize=18)
    
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    if show:
        plt.show()
    
    if num:
        figure.savefig(f'{log_path}/Metric_log_{num + 1}.jpg')
    else:
        figure.savefig(f'{log_path}/Metric_log.jpg')