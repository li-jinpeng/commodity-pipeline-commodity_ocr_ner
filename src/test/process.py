import os
from tqdm import tqdm

pid2title = {}
with open("./pid_url_title.list") as fr:
    lines = fr.readlines()
    for line in tqdm(lines):
        values = line.strip().split("\t")
        if len(values) != 3:
            continue
        pid = values[0]
        title = values[2]
        pid2title[pid] = title

output_set = set()
with open("./all_path_pid.list") as fr:
    lines = fr.readlines()
    for line in tqdm(lines):
        values = line.strip().split("\t")
        path, pid = values[:2]
        if pid not in pid2title:
            continue
        title = pid2title[pid]
        out = "\t".join([path, pid, title]) + "\n"
        output_set.add(out)

with open("./all_path_pid_title.list", 'w') as fw:
    for line in tqdm(output_set):
        fw.write(line)
