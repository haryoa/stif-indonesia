import subprocess
from .util import read_json_file
from pathlib import Path
import logging
import os
try:
    import wandb
except:
    pass
from typing import Union
import shutil
from tqdm import tqdm


MOSES_IMPLZ = 'moses/moses/bin/lmplz'
MOSES_BUILD_BINARY = 'moses/moses/bin/build_binary'
MOSES_TRAIN_PERL = 'moses/moses/scripts/training/train-model.perl'
EXTERNAL_BIN = 'moses/training-tools'
MOSES_BIN_MOSES = 'moses/moses/bin/moses'
logger = logging.getLogger("moses-rerun")
MOSES_DETOKENIZER = "moses/moses/scripts/tokenizer/detokenizer.perl"


class MosesSMTModel:

    def __init__(self, config_file:str, use_wandb=True):
        self.config_file = config_file
        self.hparams = read_json_file(config_file)
        self._smooth_variable()
        self.use_wandb = use_wandb
        if self.use_wandb:
            self._init_wandb(self.hparams['wandb_project_run'],
                             self.hparams['wandb_notes'],
                             self.hparams['wandb_name'],
                             self.hparams['wandb_tags'])

    def _init_wandb(self, wandb_project_run, wandb_notes, wandb_name, tags):
        config_wandb = dict(
            default_config=self.config_file,
            hparams=self.hparams
        )

        self.run = wandb.init(
            project=wandb_project_run,
            notes=wandb_notes,
            name=wandb_name,
            tags=tags,
            config=config_wandb,
            reinit=True
        )

    def _smooth_variable(self):
        self.root_data_pth = Path(self.hparams['data_dir'])
        self.root_output_folder = Path(self.hparams['output_working_dir'])
        self.spv_train_pth = self.root_output_folder / 'train-supervised'
        self.spv_eval = self.root_output_folder / 'eval-supervised'
        self.predicted_file = f"predict{self.hparams['target_file_type']}"
        if "semi-supervised-batch-data" in self.hparams:
            self.semi_supervised_batch_data = self.hparams['semi-supervised-batch-data']
            self.is_semi_supervised = True

    def fit(self, train_data_path: str, out_train_dir: str, absolute_lm_path: str):
        """
        Train supervised Moses
        """
        try:
            os.makedirs(out_train_dir)
        except FileExistsError:
            logger.warning(out_train_dir + " is exist")
        logger.info("Training time :).. Progressing...")
        subprocess.run([f"nice {MOSES_TRAIN_PERL} \
                        -root-dir { out_train_dir } \
                        -corpus { train_data_path } \
                        -f {self.hparams['source_file_type'].strip('.')} \
                        -e {self.hparams['target_file_type'].strip('.')} \
                        -alignment {self.hparams['moses_args']['alignment']} \
                        -reordering {self.hparams['moses_args']['reordering']} \
                        -lm 0:{self.hparams['moses_args']['moses_ngram']}:{ absolute_lm_path } \
                        -core {self.hparams['moses_args']['core_cpu']} \
                        -external-bin-dir {EXTERNAL_BIN} --mgiza"
                        ], capture_output=True, shell=True)
        logger.info("Finish Training")

    def predict(self, model_path: str, src_path: str, out_file_dir: str, use_custom_file_name=False,
                custom_file_name = 'meong.guk'):
        try:
            os.makedirs(out_file_dir)
        except FileExistsError:
            logger.warning(out_file_dir + " is exist")
        out_predict = Path(out_file_dir) / self.predicted_file if not use_custom_file_name else \
            Path(out_file_dir) / custom_file_name
        subprocess.run(
            [f"{MOSES_BIN_MOSES} -f {model_path} < {src_path} > {out_predict}"], capture_output=True, shell=True)

    def eval_bleu_moses(self, ref_file: str, evaluation_dir: str, sys_file: str):
        import sacrebleu
        try:
            os.makedirs(evaluation_dir)
        except FileExistsError:
            logger.warning(evaluation_dir + " is exist")
        subprocess.run([f"cat {ref_file} | {MOSES_DETOKENIZER} -l en > {evaluation_dir}/ref.txt"], shell=True)
        subprocess.run([f"cat {sys_file} | {MOSES_DETOKENIZER} -l en > {evaluation_dir}/sys.txt"], shell=True)
        with open(f"{evaluation_dir}/ref.txt",'r+') as file:
            refs = [file.read().split('\n')]
        with open(f"{evaluation_dir}/sys.txt",'r+') as file:
            sys = file.read().split('\n')
        bleu = sacrebleu.corpus_bleu(sys, refs)
        return bleu.score

    def run_experiments(self):
        lm_path = self.root_output_folder / 'lm'
        train_path = self.root_output_folder / 'train'
        logger.info("PERPARE KEN-LM".center(10, '='))
        binary_out = self.prepare_lm(lm_path, Path(self.root_data_pth))
        root_train_data_pth = self.root_data_pth / 'train'
        logger.info("FIT MOSES".center(10, '='))
        self.fit(str(root_train_data_pth),
                 str(train_path),
                 os.path.abspath(str(binary_out)))
        root_test_inf = self.root_data_pth / (self.hparams['data_test'] + self.hparams['source_file_type'])
        dir_out_pred = self.root_output_folder / 'evaluation'
        logger.info("PREDICT MOSES".center(10, '='))
        self.predict(str(train_path / 'model/moses.ini'), str(root_test_inf), str(dir_out_pred))
        logger.info("CALCULATE BLEU".center(10, '='))
        bleu = self.eval_bleu_moses(str(self.root_data_pth / 'test.for'),
                                    str(dir_out_pred),
                                    str(dir_out_pred / self.predicted_file))
        if self.use_wandb:
            self.run.summary['bleu_score'] = bleu
        logger.info(f"BLEU IS {bleu}".center(10, '='))

    def _makedir(self, new_dirs: Union[str, Path]):
        try:
            os.makedirs(new_dirs)
        except FileExistsError:
            logger.warning(str(new_dirs) + " is already exist")

    def _copy_data_ss(self, agg_data_dir: str):
        """
        Copy original data for semi-supervised purposes
        """
        train_data_src = f"{self.hparams['data_train']}{self.hparams['source_file_type']}"
        train_data_tgt = f"{self.hparams['data_train']}{self.hparams['target_file_type']}"
        test_data_src = f"{self.hparams['data_test']}{self.hparams['source_file_type']}"
        test_data_tgt = f"{self.hparams['data_test']}{self.hparams['target_file_type']}"
        aggregate_src = [train_data_src, train_data_tgt, test_data_src, test_data_tgt]
        logger.info(f"List of aggregate {aggregate_src}")
        agg_data_dir = Path(agg_data_dir)
        for src in aggregate_src:
            logger.info(f"Copying {str(self.root_data_pth / src) } into "
                        f" {str(agg_data_dir / src)}")
            shutil.copy(str(self.root_data_pth / src), str(agg_data_dir / src))

    def run_moses_experiment(self, output_folder: Path, data_path: Path):
        lm_path = output_folder / 'lm'
        train_path = output_folder / 'train'
        logger.info("PERPARE KEN-LM".center(10, '='))
        binary_out = self.prepare_lm(lm_path, Path(data_path))
        root_train_data_pth = data_path / self.hparams['data_train']
        logger.info("FIT MOSES".center(10, '='))
        self.fit(str(root_train_data_pth),
                 str(train_path),
                 os.path.abspath(str(binary_out)))
        root_test_inf = data_path / (self.hparams['data_test'] + self.hparams['source_file_type'])
        dir_out_pred = output_folder / 'evaluation'
        logger.info("PREDICT MOSES".center(10, '='))
        self.predict(str(train_path / 'model/moses.ini'), str(root_test_inf), str(dir_out_pred))
        logger.info("CALCULATE BLEU".center(10, '='))
        bleu = self.eval_bleu_moses(str(data_path / 'test.for'),
                                    str(dir_out_pred),
                                    str(dir_out_pred / self.predicted_file))
        return bleu, train_path, dir_out_pred

    def run_semi_supervised(self):
        """
        Run semi supervised here.
        Save the best one and latest run model
        """
        # Data for being used on current batch
        produced_tgt_data = self.root_output_folder / 'produced_tgt_data'  # Act as collection of tgt_data
        best_model_dir = self.root_output_folder / 'best_model_dir'  # Best model place
        current_model_dir = self.root_output_folder / 'current_model_dir'  # Runnning model
        agg_data_dir = self.root_output_folder / 'agg_data'   # Aggregator data
        self._makedir(agg_data_dir)
        self._makedir(best_model_dir)

        bleu_best = -1
        bleu_best_run = 'supervised'
        logger.info(f"Copy data from {self.hparams['data_train']} into {agg_data_dir}")
        self._copy_data_ss(agg_data_dir)
        list_batch_ss = os.listdir(self.hparams['semi-supervised-batch-data'])

        # Go on semi supervised
        for i, crnt_batch in tqdm(enumerate(list_batch_ss)):
            self._makedir(current_model_dir)
            logger.info(f"Running iter-{i}")
            logger.info(f"Current iter {i} will predict {crnt_batch}")
            bleu, train_path, dir_out_pred = self.run_moses_experiment(current_model_dir, agg_data_dir)
            if bleu > bleu_best:
                bleu_best_run = i
                self._copy_best_model(current_model_dir, best_model_dir)
                bleu_best = bleu
            logger.info(f"CURRENT BLEU FOR {i} is {bleu}")
            logger.info(f"LEADERBOARD BLEU BEST = {bleu_best} is {bleu_best_run}")

            pred_batch_output = f'pred_{i}_' + crnt_batch + self.hparams['target_file_type']
            logger.info(f"Predicting {pred_batch_output}")

            self.predict(str(train_path / 'model/moses.ini'),
                         str(Path(self.hparams['semi-supervised-batch-data']) / crnt_batch),
                         str(produced_tgt_data), True,
                         pred_batch_output)

            # copy predicted batch to run_data
            logger.info(f"Append predicted into {agg_data_dir}")
            self._append_batch(agg_data_dir / (self.hparams['data_train'] + self.hparams['source_file_type']),
                               agg_data_dir / (self.hparams['data_train'] + self.hparams['target_file_type']),
                               Path(self.hparams['semi-supervised-batch-data']) / crnt_batch,
                               produced_tgt_data / pred_batch_output)

            # Remove current model directory as moses get 'weird'
            logger.info(f"Destroy {current_model_dir}")
            shutil.rmtree(current_model_dir)
            if self.use_wandb:
                self.run.log({'bleu': bleu}, step=i)

        if self.use_wandb:
            self.run.summary['bleu_score'] = bleu
            self.run.summary['bleu_best_run'] = bleu_best_run

    def _copy_best_model(self, current_model_path, best_model_path):
        try:
            shutil.rmtree(best_model_path)
        except FileNotFoundError:
            pass
        logger.info(f"Copy from {current_model_path} into {best_model_path}")
        shutil.copytree(current_model_path, best_model_path)

    def _append_batch(self, run_informal, run_formal, batch_informal, batch_formal):
        with open(run_informal, 'a') as file_inf:
            with open(batch_informal, 'r') as b_inf:
                for line in b_inf:
                    file_inf.write(line)

        with open(run_formal, 'a') as file_for:
            with open(batch_formal, 'r') as b_for:
                for line in b_for:
                    file_for.write(line)

    def prepare_lm(self, out_lm_dir: Path, train_data_dir: Path) -> str:
        """
        Prepare KenLM stuffs
        """
        arpa_out = out_lm_dir / ('blm/data.arpa' + self.hparams['target_file_type'])
        binary_out = out_lm_dir / ('blm/data.blm' + self.hparams['target_file_type'])
        import os
        try:
            os.makedirs(str(out_lm_dir / 'blm'))
        except FileExistsError:
            logger.warning(str(out_lm_dir / 'blm') + " is exist")
        data_train_pth = str(train_data_dir / (self.hparams['data_train'] + self.hparams['target_file_type']))

        subprocess.run([f"{MOSES_IMPLZ} -o {self.hparams['moses_args']['moses_ngram']}"
                        f" < {data_train_pth} "
                        f" > {arpa_out} "],
                       shell=True)
        subprocess.run([f"{MOSES_BUILD_BINARY} {arpa_out} {binary_out}"], shell=True)
        return binary_out

