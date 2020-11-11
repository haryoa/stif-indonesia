import subprocess

"""
use .src and .tgt as the file extensions
"""


def run_pipeline(base_model_name="working/train/model/moses.ini", base_data_name = "data", 
                batch_data_dir = "data/unlabeled_batches/", batch_predictions_dir = "out_predict/batch_predicts_semi_fixed/", 
                logs_dir = "logs/", batch_name="batch", iters=10, eval_dir = "out_predict/batch_eval_semi_fixed/"):
    model = base_model_name
    data = base_data_name
    
    gold_prediction_path = f"{eval_dir}data_test_supervised.tgt"
    gold_result_path = f"{eval_dir}eval"

    predict_moses(f"working/train/model/moses.ini", "data/data_test.src", gold_prediction_path)
    bleu_calculate(gold_prediction_path, "data/data_test.tgt", f"{gold_result_path}_semi_sup_fixed.txt")
    for i in range(iters):
        src_dir = f"{batch_data_dir}{batch_name}{i}.inf"
        print(f"Batch {i}")
        completed_process = subprocess.run(
            [f"moses/bin/moses -f {model} < {src_dir}"], capture_output=True, shell=True)
        out = open(f"{batch_predictions_dir}predict{i}.for", "wb")
        err = open(f"{logs_dir}log{i}.txt", "wb")
        out.write(completed_process.stdout)
        err.write(completed_process.stderr)
        out.close()
        err.close()
        with open(f"data/{data}.src", "a") as src_pseudo_data, open(src_dir, "r") as input_sentences:
            src_pseudo_data.write(input_sentences.read())
        with open(f"data/{data}.tgt", "ab") as tgt_pseudo_data:
            tgt_pseudo_data.write(completed_process.stdout)
        print(f"Finished Batch {i} Predictions, training LM...")
        subprocess.run([f"moses/bin/lmplz -o 3 < data/{data}.tgt > data/{data}.arpa.tgt"], shell=True)
        subprocess.run(
            [f"moses/bin/build_binary data/{data}.arpa.tgt data/{data}.blm.tgt"], shell=True)
        print(f"Training Batch {i} Moses Model")
        train_process = subprocess.run([f"nice moses/scripts/training/train-model.perl \
                                            -root-dir working/train \
                                            -corpus data/{data} \
                                            -f src \
                                            -e tgt \
                                            -alignment grow-diag-final-and \
                                            -reordering msd-bidirectional-fe \
                                            -lm 0:3:/home/tatag/moses/data/{data}.blm.tgt \
                                            -core 2 \
                                            -external-bin-dir moses/tools"], capture_output=True, shell=True)
        with open(f"{logs_dir}second_try/log{i}.txt","ab") as log:
            try:
                log.write(train_process.stderr)
            except:
                pass
        
        gold_prediction_path = f"{eval_dir}data_test_predict{i}.tgt"
        gold_result_path = f"{eval_dir}eval"

        predict_moses(f"working/train/model/moses.ini", "data/data_test.src", gold_prediction_path)
        bleu_calculate(gold_prediction_path, "data/data_test.tgt", f"{gold_result_path}_semi_sup_fixed.txt")

        subprocess.run([f"cp -R working/train/model saved_models/semi-sup-fixed/batch_{i}"], shell=True)


def evaluate_models(model_path="saved_models/", n_models = 10, test_tgt_path = "data/data_test.tgt",
                     test_src_path = "data/data_test.src", eval_file = "out_predict/batch_eval/eval", 
                     prediction_path = "out_predict/batch_eval/data_test_predict"):
    for i in range (n_models):
        predict_moses(f"{model_path}batch_{i}_moses.ini", test_src_path, f"{prediction_path}{i}.tgt")
        bleu_calculate(f"{prediction_path}{i}.tgt", test_tgt_path, f"{eval_file}{i}.txt")
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
    run_pipeline(batch_data_dir="data/batches_100/txt/", iters=100)
