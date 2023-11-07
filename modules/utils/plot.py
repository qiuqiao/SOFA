import matplotlib.pyplot as plt
import numpy as np


def plot_for_test(
        melspec,
        ph_seq,
        ph_time,
        frame_confidence,
        ph_frame_prob,
        ph_frame_id_gt,
        edge_prob,
):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(melspec[0],
               origin="lower",
               aspect="auto")

    for i, t in enumerate(ph_time):
        ax1.axvline(t, color="r", linewidth=1)
        if ph_seq[i] != "<EMPTY>":
            if i % 2:
                ax1.text(ph_time[i], melspec.shape[-2], ph_seq[i], fontsize=11)
            else:
                ax1.text(ph_time[i], melspec.shape[-2] - 5, ph_seq[i], fontsize=11, color="white")

    x = np.arange(melspec.shape[-1])
    ax1.plot(x, frame_confidence * melspec.shape[-2], color="black", linewidth=1, alpha=0.6)
    ax1.fill_between(x, frame_confidence * melspec.shape[-2], color="black", alpha=0.3)

    ax2.imshow(ph_frame_prob.T,
               origin="lower",
               aspect="auto",
               interpolation='nearest')

    ax2.plot(x, ph_frame_id_gt, color="red", linewidth=1.5)
    # ax2.scatter(x, ph_frame_id_gt, s=5, marker='s', color="red")

    ax2.plot(x, edge_prob * ph_frame_prob.shape[-1], color="black", linewidth=1)
    ax2.fill_between(x, edge_prob * ph_frame_prob.shape[-1], color="black", alpha=0.3)

    fig.set_size_inches(15, 10)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.subplots_adjust(hspace=0)

    return fig
