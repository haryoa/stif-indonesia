import click
from .main_moses import MosesSMTModel
from .logging import CustomLogger


sup_experiments = [
    'experiment-config/00001_default_supervised_config.json'
]


experiment_semi_supervised = [
    'experiment-config/00002_default_semi_supervised_config.json',
]


def do_experiment(exp):
    moses_model = MosesSMTModel(exp, use_wandb=False)
    moses_model.run_experiments()

def do_semi_supervised_experiment(exp):
    moses_model = MosesSMTModel(exp, use_wandb=False)
    moses_model.run_semi_supervised()


@click.command()
@click.option('--exp-scenario', help='possible "supervised" or "semi-supervised"')
def main(exp_scenario: str):
    """
    To customize your needs, add your experiment config and put it to the 'sup_experiments' or 'experiment_semi_supervised'
    """
    CustomLogger().create_logger('moses-rerun', log_file='log.log', alay=True)
    if exp_scenario == 'supervised':
        for exp in sup_experiments:
            do_experiment(exp)
    if exp_scenario == 'semi-supervised':
        for exp in experiment_semi_supervised:
            do_semi_supervised_experiment(exp)

if __name__ == '__main__':
    main()
