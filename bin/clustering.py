# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script is designed to cluster speaker embeddings and generate RTTM result files as output.
"""

import os
import sys
import argparse
import numpy as np
import shutil

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from utils.tools import build, load_yaml_config

parser = argparse.ArgumentParser(description='Cluster embeddings and output rttm files')
parser.add_argument('--conf', default="config/build_benchmark.yaml", help='Config file')
parser.add_argument('--wavs', default="pretrained/speech_campplus_sv_zh-cn_16k-common/embeddings/wav_list.txt", help='Wav list file')
parser.add_argument('--audio_embs_dir', default="pretrained/speech_campplus_sv_zh-cn_16k-common/embeddings", type=str, help='Embedding dir')
parser.add_argument('--output_dir', default="./", type=str, help='output dir')

def audio_only_func(local_wav_list, audio_embs_dir, output_dir, config):
    cluster = build('cluster', config)
    embeddings = []
    for wav_file in local_wav_list:
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        embs_file = os.path.join(audio_embs_dir, rec_id + '.npy')
        embedding = np.load(embs_file)
        embeddings.append(embedding)

    # cluster
    embeddings = np.array(embeddings)
    labels, top_indices_dict = cluster(embeddings, 10)

    cluster_output = os.path.join(output_dir, "cluster_output")
    os.makedirs(cluster_output, exist_ok=True)

    # output cluster result
    for cluster_id, indices in enumerate(top_indices_dict.values()):
        sub_cluster_output = os.path.join(cluster_output, f"cluster_{cluster_id}")
        os.makedirs(sub_cluster_output, exist_ok=True)
        for count, i in enumerate(indices[::-1].tolist()):
            source_path = local_wav_list[i[0]]
            sourece_id = os.path.basename(source_path).rsplit('.', 1)[0]
            target_path = os.path.join(sub_cluster_output, f"top_{count}_{sourece_id}.wav")
            shutil.copy(source_path, target_path)

    with open(os.path.join(output_dir, "cluster_result.txt"), 'w') as f:
        for wav_file, cluster_id in zip(local_wav_list, labels):
            f.write(f"{wav_file} {cluster_id}\n")

def main():
    args = parser.parse_args()
    with open(args.wavs,'r') as f:
        wav_list = [i.strip() for i in f.readlines()]
    wav_list.sort()

    os.makedirs(args.output_dir, exist_ok=True)
    print("[INFO] Start clustering...")
    config = load_yaml_config(args.conf)
    audio_only_func(wav_list, args.audio_embs_dir, args.output_dir, config)

if __name__ == "__main__":
    main()
