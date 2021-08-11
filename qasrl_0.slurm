#! /bin/sh

#SBATCH --job-name=qasrl_0
#SBATCH --output=qasrl_0.out
#SBATCH --error=qasrl_0.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mail-user=rubenw@mail.tau.ac.il
#SBATCH --mail-type=END

srun python -m allennlp.run predict ./data/qasrl_parser_elmo /home/yandex/AMNLP2021/rubenw/project/s2e-coref/data_by_sentences_out/data_sentence_${1}.jsonl --include-package nrl --predictor qasrl_parser --output-file /home/yandex/AMNLP2021/rubenw/project/nrl-qasrl/data_by_sentences_qasrl/qasrl_${1}.jsonl --silent --cuda-device 0
