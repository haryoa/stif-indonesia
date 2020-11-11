import subprocess
import argparse

DEFAULT = 0
NO_PUNCTUATION = 1
LOWERCASED = 2
NO_PUNCTUATION_LOWERCASED = 3

def init_supervised_model(scenario):
    annotated_data_dir, logs_dir, eval_dir, model_save_path, _ = get_variables(scenario)

    subprocess.run([f"cp {annotated_data_dir}/data.src data/data_raw.src"], shell=True)
    subprocess.run([f"cp {annotated_data_dir}/test.inf data/data_test.src"], shell=True)

    subprocess.run([f"cp {annotated_data_dir}/data.tgt data/data_raw.tgt"], shell=True)
    subprocess.run([f"cp {annotated_data_dir}/test.for data/data_test.tgt"], shell=True)
    
    subprocess.run(["moses/scripts/tokenizer/tokenizer.perl -l en < data/data_raw.src > data/data.src"], shell=True)
    subprocess.run(["moses/scripts/tokenizer/tokenizer.perl -l en < data/data_raw.tgt > data/data.tgt"], shell=True)

    subprocess.run(["moses/bin/lmplz -o 3 < data/data.tgt > data/data.arpa.tgt"], shell=True)
    subprocess.run(
            ["moses/bin/build_binary data/data.arpa.tgt data/data.blm.tgt"], shell=True)
    train_process = subprocess.run([f"nice moses/scripts/training/train-model.perl \
                                            -root-dir working/train \
                                            -corpus data/data \
                                            -f src \
                                            -e tgt \
                                            -alignment grow-diag-final-and \
                                            -reordering msd-bidirectional-fe \
                                            -lm 0:3:/home/tatag/moses/data/data.blm.tgt \
                                            -core 2 \
                                            -external-bin-dir moses/tools"], capture_output=True, shell=True)
    with open(f"{logs_dir}/init_model.txt","ab") as log:
        try:
            log.write(train_process.stderr)
        except:
            pass

    gold_hyp_path = f"{eval_dir}/hyp_test_supervised.tgt"
    gold_eval_path = f"{eval_dir}/bleu_eval.txt"

    predict_moses(f"working/train/model/moses.ini", "data/data_test.src", gold_hyp_path)
    bleu_calculate(gold_hyp_path, "data/data_test.tgt", f"{gold_eval_path}")

    subprocess.run([f"cp -R working/train/model {model_save_path}"], shell=True)

def iterate_semi_supervised(scenario):
    _, logs_dir, eval_dir, model_save_path, batch_data_dir = get_variables(scenario)
    for i in range(100):
        src_dir = f"{batch_data_dir}/txt/batch{i}.inf"
        print(f"Batch {i}")
        completed_process = subprocess.run(
            [f"moses/bin/moses -f working/train/model/moses.ini < {src_dir}"], capture_output=True, shell=True)
        out = open(f"{batch_data_dir}/pseudo/predict{i}.for", "wb")
        err = open(f"{logs_dir}/log{i}.txt", "wb")
        out.write(completed_process.stdout)
        err.write(completed_process.stderr)
        out.close()
        err.close()
        with open(f"data/data.src", "a") as src_pseudo_data, open(src_dir, "r") as input_sentences:
            src_pseudo_data.write(input_sentences.read())
        with open(f"data/data.tgt", "ab") as tgt_pseudo_data:
            tgt_pseudo_data.write(completed_process.stdout)
        print(f"Finished Batch {i} Predictions, training LM...")
        subprocess.run([f"moses/bin/lmplz -o 3 < data/data.tgt > data/data.arpa.tgt"], shell=True)
        subprocess.run(
            [f"moses/bin/build_binary data/data.arpa.tgt data/data.blm.tgt"], shell=True)
        print(f"Training Batch {i} Moses Model")
        train_process = subprocess.run([f"nice moses/scripts/training/train-model.perl \
                                            -root-dir working/train \
                                            -corpus data/data \
                                            -f src \
                                            -e tgt \
                                            -alignment grow-diag-final-and \
                                            -reordering msd-bidirectional-fe \
                                            -lm 0:3:/home/tatag/moses/data/data.blm.tgt \
                                            -core 2 \
                                            -external-bin-dir moses/tools"], capture_output=True, shell=True)
        with open(f"{logs_dir}/log{i}.txt","ab") as log:
            try:
                log.write(train_process.stderr)
            except:
                pass
        
        gold_hyp_path = f"{eval_dir}/hyp_test_{i}.tgt"
        gold_eval_path = f"{eval_dir}/bleu_eval.txt"

        predict_moses(f"working/train/model/moses.ini", "data/data_test.src", gold_hyp_path)
        bleu_calculate(gold_hyp_path, "data/data_test.tgt", f"{gold_eval_path}")

        subprocess.run([f"cp -R working/train/model {model_save_path}"], shell=True)

def get_variables(scenario):
    if scenario == DEFAULT:
        annotated_data_dir = "data/v1/default"
        logs_dir = "logs/semi-sup/default"
        eval_dir = "evaluation/semi_supervised/default"
        model_save_path = "models/semi-supervised/default"
        batch_data_dir = "data/batches/default"
    elif scenario == NO_PUNCTUATION:
        annotated_data_dir = "data/v1/0_lowercase_0_punctuation"
        logs_dir = "logs/semi-sup/0_lowercase_0_punctuation"
        eval_dir = "evaluation/semi_supervised/0_lowercase_0_punctuation"
        model_save_path = "models/semi-supervised/0_lowercase_0_punctuation"
        batch_data_dir = "data/batches/0_lowercase_0_punctuation"
    elif scenario == LOWERCASED:
        annotated_data_dir = "data/v1/1_lowercase_1_punctuation"
        logs_dir = "logs/semi-sup/1_lowercase_1_punctuation"
        eval_dir = "evaluation/semi_supervised/1_lowercase_1_punctuation"
        model_save_path = "models/semi-supervised/1_lowercase_1_punctuation"
        batch_data_dir = "data/batches/1_lowercase_1_punctuation"
    elif scenario == NO_PUNCTUATION_LOWERCASED:
        annotated_data_dir = "data/v1/1_lowercase_0_punctuation"
        logs_dir = "logs/semi-sup/1_lowercase_0_punctuation"
        eval_dir = "evaluation/semi_supervised/1_lowercase_0_punctuation"
        model_save_path = "models/semi-supervised/1_lowercase_0_punctuation"
        batch_data_dir = "data/batches/1_lowercase_0_punctuation"
    else:
        return
    return annotated_data_dir, logs_dir, eval_dir, model_save_path, batch_data_dir
    
def bleu_calculate(prediction_path, test_path, eval_file):
    completed_process = subprocess.run([f"moses/scripts/generic/multi-bleu.perl -lc {test_path} \
        < {prediction_path}"], capture_output=True, shell=True)
    with open(f"{eval_file}", "ab") as out:
        out.write(completed_process.stdout)
    
def predict_moses(model_path, src_path, predict_path):
    completed_process = subprocess.run(
            [f"moses/bin/moses -f {model_path} < {src_path}"], capture_output=True, shell=True)
    out = open(f"{predict_path}", "wb")
    err = open(f"{predict_path}.log", "wb")
    out.write(completed_process.stdout)
    err.write(completed_process.stderr)
    out.close()
    err.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", 
        help="0 = default,1 = no_punctuation,2 = lowercased, 3 = no_p_lowercased", type=int)
    args = parser.parse_args()
    scenario = args.scenario

    init_supervised_model(scenario)
    iterate_semi_supervised(scenario)