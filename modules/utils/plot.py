import matplotlib.pyplot as plt
import numpy as np


def plot_for_valid(
    input_feature,
    ph_seq,
    ph_intervals,
    frame_confidence,
    ph_frame_prob,
    ph_frame_id_gt,
):
    fig, (ax1, ax2) = plt.subplots(2)

    # plot1
    # melspec
    melspec = input_feature[:, :-3, :]
    ax1.imshow(melspec[0], origin="lower", aspect="auto")

    # ph_seq
    ph_seq = [i.split("/")[-1] for i in ph_seq]
    for i, interval in enumerate(ph_intervals):
        if i == 0 or (i > 0 and ph_intervals[i - 1, 1] != interval[0]):
            if interval[0] > 0:
                ax1.axvline(interval[0], color="r", linewidth=1)
        if interval[1] < melspec.shape[-1]:
            ax1.axvline(interval[1], color="r", linewidth=1)
        if ph_seq[i] != "SP":
            if i % 2:
                ax1.text(
                    (interval[0] + interval[1]) / 2
                    - len(ph_seq[i]) * melspec.shape[-1] / 275,
                    melspec.shape[-2] + 1,
                    ph_seq[i],
                    fontsize=11,
                    color="black",
                )
            else:
                ax1.text(
                    (interval[0] + interval[1]) / 2
                    - len(ph_seq[i]) * melspec.shape[-1] / 275,
                    melspec.shape[-2] - 6,
                    ph_seq[i],
                    fontsize=11,
                    color="white",
                )

    # f0,uv,energy
    diff_midi, uv, energy = (
        input_feature[0, -3, :],
        input_feature[0, -2, :],
        input_feature[0, -1, :],
    )
    midi_change_rate = np.abs(diff_midi)

    x = np.arange(melspec.shape[-1])

    ax1.plot(
        x,
        (uv) * melspec.shape[-2],
        color="black",
        linewidth=0.1,
        alpha=0.6,
    )
    ax1.fill_between(x, (uv) * melspec.shape[-2], color="black", alpha=0.3)

    ax1.plot(
        x,
        (midi_change_rate / 6) * melspec.shape[-2],
        color="blue",
        linewidth=1,
        alpha=0.6,
    )
    # ax1.fill_between(
    #     x, (midi_change_rate / 6) * melspec.shape[-2], color="blue", alpha=0.3
    # )

    ax1.plot(
        x,
        (energy) * melspec.shape[-2],
        color="purple",
        linewidth=1,
        alpha=0.6,
    )
    # ax1.fill_between(x, (energy) * melspec.shape[-2], color="yellow", alpha=0.3)

    # plot2
    # pred_prob
    ax2.imshow(
        ph_frame_prob.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        # vmin=0,
        # vmax=1,
    )

    # pred_label
    ax2.plot(x, ph_frame_id_gt, color="red", linewidth=1.5)

    # confidence
    ax2.plot(
        x,
        frame_confidence * ph_frame_prob.shape[-1],
        color="black",
        linewidth=1,
        alpha=0.6,
    )
    ax2.fill_between(
        x, frame_confidence * ph_frame_prob.shape[-1], color="black", alpha=0.3
    )

    fig.set_size_inches(11, 6)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    return fig
