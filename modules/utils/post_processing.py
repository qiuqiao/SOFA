MIN_SP_LENGTH = 0.1
SP_MERGE_LENGTH = 0.3


def add_SP(word_seq, word_intervals, wav_length):
    word_seq_res = []
    word_intervals_res = []
    if len(word_seq) == 0:
        word_seq_res.append("SP")
        word_intervals_res.append([0, wav_length])
        return word_seq_res, word_intervals_res

    word_seq_res.append("SP")
    word_intervals_res.append([0, word_intervals[0, 0]])
    for word, (start, end) in zip(word_seq, word_intervals):
        if word_intervals_res[-1][1] < start:
            word_seq_res.append("SP")
            word_intervals_res.append([word_intervals_res[-1][1], start])
        word_seq_res.append(word)
        word_intervals_res.append([start, end])
    if word_intervals_res[-1][1] < wav_length:
        word_seq_res.append("SP")
        word_intervals_res.append([word_intervals_res[-1][1], wav_length])
    if word_intervals[0, 0] <= 0:
        word_seq_res = word_seq_res[1:]
        word_intervals_res = word_intervals_res[1:]

    return word_seq_res, word_intervals_res


def fill_small_gaps(word_seq, word_intervals, wav_length):
    if word_intervals[0, 0] > 0:
        if word_intervals[0, 0] < MIN_SP_LENGTH:
            word_intervals[0, 0] = 0

    for idx in range(len(word_seq) - 1):
        if word_intervals[idx, 1] < word_intervals[idx + 1, 0]:
            if word_intervals[idx + 1, 0] - word_intervals[idx, 1] < SP_MERGE_LENGTH:
                if word_seq[idx] == "AP":
                    if word_seq[idx + 1] == "AP":
                        # 情况1：gap的左右都是AP
                        mean = (word_intervals[idx, 1] + word_intervals[idx + 1, 0]) / 2
                        word_intervals[idx, 1] = mean
                        word_intervals[idx + 1, 0] = mean
                    else:
                        # 情况2：只有左边是AP
                        word_intervals[idx, 1] = word_intervals[idx + 1, 0]
                elif word_seq[idx + 1] == "AP":
                    # 情况3：只有右边是AP
                    word_intervals[idx + 1, 0] = word_intervals[idx, 1]
                else:
                    # 情况4：gap的左右都不是AP
                    if (
                            word_intervals[idx + 1, 0] - word_intervals[idx, 1]
                            < MIN_SP_LENGTH
                    ):
                        mean = (word_intervals[idx, 1] + word_intervals[idx + 1, 0]) / 2
                        word_intervals[idx, 1] = mean
                        word_intervals[idx + 1, 0] = mean

    if word_intervals[-1, 1] < wav_length:
        if wav_length - word_intervals[-1, 1] < MIN_SP_LENGTH:
            word_intervals[-1, 1] = wav_length

    return word_seq, word_intervals


def post_processing(predictions):
    print("Post-processing...")

    res = []
    error_log = []
    for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
    ) in predictions:
        try:
            # fill small gaps
            word_seq, word_intervals = fill_small_gaps(
                word_seq, word_intervals, wav_length
            )
            ph_seq, ph_intervals = fill_small_gaps(ph_seq, ph_intervals, wav_length)
            # add SP
            word_seq, word_intervals = add_SP(word_seq, word_intervals, wav_length)
            ph_seq, ph_intervals = add_SP(ph_seq, ph_intervals, wav_length)

            res.append(
                [
                    wav_path,
                    wav_length,
                    confidence,
                    ph_seq,
                    ph_intervals,
                    word_seq,
                    word_intervals,
                ]
            )
        except Exception as e:
            error_log.append([wav_path, e])
    return res, error_log
