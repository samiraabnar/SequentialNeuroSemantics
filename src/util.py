import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style
import tensorflow as tf
import os

def plot(predictions, targets, train_step, words, plot_size,FLAGS):
    style.use('seaborn')
    font_dict = {'family': 'serif',
                                 'color':'darkred',
                                  'size':5}

    fig = plt.figure(figsize=(plot_size, plot_size))
    gs = gridspec.GridSpec(plot_size, plot_size)
    gs.update(wspace=0.5, hspace=0.5)

    #print(len(predictions))
    #print(len(targets))
    for i, p_t in enumerate(zip(predictions, targets)):
        p, t = p_t
        ax = plt.subplot(gs[i])
        sorted_voxel_indexes = np.argsort(t)
        time_steps = np.arange(len(sorted_voxel_indexes))
        ax.plot(time_steps, t[sorted_voxel_indexes], time_steps, p[sorted_voxel_indexes], linewidth=0.5)
        ax.axis('off')
        #ax.set_xlabel('voxels (sorted based on label)')
        #ax.set_ylabel('label and predicted')
        ax.set_title(words[i], fontdict=font_dict)
        ax.grid(False)

    plots_path = os.path.join(FLAGS.log_root, 'plots')
    if not os.path.exists(plots_path): os.makedirs(plots_path)

    plt.savefig(plots_path + '/{}.png'.format(str(train_step).zfill(3)),
                bbox_inches='tight', dpi=300)

    tf.logging.info("output plots for test data: " +
                    plots_path + '/{}.png'.format(str(train_step).zfill(3)))

    plt.close(fig)

    return fig

def plot_all2one(predictions, targets, words,train_step, plot_size,FLAGS):
    style.use('seaborn')
    font_dict = {'family': 'serif',
                                 'color':'darkred',
                                  'size':4}

    fig = plt.figure(figsize=(plot_size, plot_size))
    gs = gridspec.GridSpec(plot_size, plot_size)
    gs.update(wspace=2.0, hspace=2.0)

    #print(len(predictions))
    #print(len(targets))
    for i, p_t in enumerate(zip(targets, words)):
        target, word = p_t
        print(word)
        ax = plt.subplot(gs[i])
        sorted_voxel_indexes = np.argsort(target)
        time_steps = np.arange(len(sorted_voxel_indexes))
        ax.plot(time_steps, target[sorted_voxel_indexes])

        for k in np.arange(len(targets)):
            if k != i:
                ax.plot(time_steps, targets[k][sorted_voxel_indexes], linewidth=0.5)
            
        ax.axis('off')
        #ax.set_xlabel('voxels (sorted based on label)')
        #ax.set_ylabel('label and predicted')
        
        ax.grid(False)
        ax.set_title(words[i], fontdict=font_dict)
    plots_path = os.path.join(FLAGS.log_root, 'plots')
    if not os.path.exists(plots_path): os.makedirs(plots_path)

    plt.savefig(plots_path + '/all2one{}.png'.format(str(train_step).zfill(3)),
                bbox_inches='tight', dpi=300)

    tf.logging.info("output plots for test data: " +
                    plots_path + '/{}.png'.format(str(train_step).zfill(3)))

    plt.close(fig)

    return fig

def is_float(text):
    try:
        float(text)
        # check for nan/infinity etc.
        if text.isalpha():
            return False
        return True
    except ValueError:
        return False