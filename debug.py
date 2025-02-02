import json
import sys

from allennlp.commands import main

model_file = "/home/yandex/AMNLP2021/rubenw/project/nrl-qasrl/data/qasrl_parser_elmo"
input_file = "/home/yandex/AMNLP2021/rubenw/project/nrl-qasrl/data_sentence_end_3.jsonl"
out_file = "/home/yandex/AMNLP2021/rubenw/project/s2e-coref/data_by_sentences_out/qasrl_end_3.jsonl"
# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
# shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp.run",  # command name, not used by main
    "predict",
    model_file,
    input_file,
    "--include-package", "nrl",
    "--predictor","qasrl_parser", "--output-file", out_file,"--silent","--cuda-device","0",
]


main()