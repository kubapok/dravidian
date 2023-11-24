import pandas as pd


def get_data_dict():
    r = pd.read_csv('../data/kannada_final.tsv',sep=',')


    spans = [a.strip('[]').split(', ') for a in r['spans'].to_list()]
    spans = [[int(b) for b in s ] for s in spans]
    texts = r['text'].to_list()
    out_tokens_list = list()
    out_labels_list = list()


    for i in range(len(spans)):
        not_off_begin = texts[i][:spans[i][0]]
        off = texts[i][spans[i][0]:spans[i][-1]]
        not_off_end = texts[i][spans[i][-1]:]
        out_tokens = not_off_begin.split() + off.split() + not_off_end.split()
        out_labels = ['0' for _ in range(len(not_off_begin.split()))] + ['1' for _ in range(len(off.split()))] + ['0' for _ in range(len(not_off_end.split()))]
        out_tokens_list.append(out_tokens)
        out_labels_list.append(out_labels)
    return out_tokens_list, out_labels_list


