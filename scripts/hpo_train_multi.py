import optuna
from scripts.train_simclr import main, parse_args
import argparse
from types import SimpleNamespace



ap = argparse.ArgumentParser()
ap.add_argument("--optuna_study_name", type=str, default="study_name")
ap.add_argument("--optuna_db", type=str, default="optuna.db")
ap.add_argument("--tune_spectral_loss_coeff", action="store_true")
ap.add_argument("--tune_target_alpha", action="store_true")
ap.add_argument("--tune_temperature", action="store_true")
ap.add_argument("--random_sampler", action="store_true")

args = parse_args(ap=ap)

if args.grid_search:
    search_space = {"tau": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "spectral_loss_coeff": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                    "target_alpha": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]}

    study = optuna.create_study(
        storage='sqlite:///{}'.format(args.optuna_db),
        study_name=args.optuna_study_name,
        load_if_exists=True,
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.RandomSampler(search_space)
    )
else:
        study = optuna.create_study(
        storage='sqlite:///{}'.format(args.optuna_db),
        study_name=args.optuna_study_name,
        load_if_exists=True,
        directions=['maximize', 'maximize']
    )

def objective(trial):
    # spectral_loss_warmup_epochs = trial.suggest_int("spectral_loss_warmup_epochs", 0, 20, step=5)
    if args.tune_temperature:
        if args.random_sampler:
            args.tau = trial.suggest_categorical("tau", search_space["tau"])
        else:
            args.tau = trial.suggest_float("tau", 0.05, 0.5, step=0.05)
        args.tag = "tuning_temperature"
        print(f"tau: {args.tau}")
    if args.tune_spectral_loss_coeff:
        if args.random_sampler:
            args.spectral_loss_coeff = trial.suggest_categorical("spectral_loss_coeff", search_space["spectral_loss_coeff"])
        else:
            args.spectral_loss_coeff = trial.suggest_float("spectral_loss_coeff", 0.0, 2.0, step=0.05)
        args.tag = "tuning_spectral_loss_coeff"
        print(f"spectral_loss_coeff: {args.spectral_loss_coeff}")
    if args.tune_target_alpha:
        assert args.skip_alpha == False, "Cannot tune target_alpha if skip_alpha is True"
        assert args.spectral_loss_coeff != 0.0, "Cannot tune target_alpha if spectral_loss_coeff is 0.0"
        if args.random_sampler:
            args.target_alpha = trial.suggest_categorical("target_alpha", search_space["target_alpha"])
        else:
            args.target_alpha = trial.suggest_float("target_alpha", 0.0, 2.0, step=0.05)
        args.tag = "tuning_alpha"
        print(f"target_alpha: {args.target_alpha}")
    
    # kwargs = {"imagenet_root": args.imagenet_root, "epochs": args.epochs, "batch_size": args.batch_size,
    #         "img_size": args.image_size, "tau": args.tau, "lr": args.lr, "wd": args.wd, "workers": args.workers, "accum_steps": args.accum_steps,
    #         "warmup_epochs": args.warmup_epochs, "amp": args.amp, "grad_clip": args.grad_clip, "limit_train": args.limit_train, "limit_val": args.limit_val,
    #         "log_every": args.log_every, "save_dir": args.save_dir, "skip_knn": args.skip_knn, "skip_alpha": args.skip_alpha, "skip_neural_ev": args.skip_neural_ev,
    #         "skip_linear_probe": args.skip_linear_probe, "skip_pr": args.skip_pr, "lp_epochs": args.lp_epochs, "lp_lr": args.lp_lr,
    #         "lp_wd": args.lp_wd, "spectral_loss_coeff": args.spectral_loss_coeff, "spectral_loss_warmup_epochs": args.spectral_loss_warmup_epochs,
    #         "neural_ev_layer": args.neural_ev_layer, "neural_data_dir": args.neural_data_dir, "seed": args.seed, "single": False, "target_alpha": args.target_alpha, "command": False, "eval_every": 10, }

    # f_ev, r_ev = main(**kwargs)

    # f_ev, r_ev = main(SimpleNamespace(**kwargs), skip_eval = True)
    f_ev, r_ev = main(args)
    return f_ev, r_ev



study.optimize(objective, n_trials=1)