import optuna
from scripts.train_simclr_func import main
import argparse

def parse_args():
    ap = argparse.ArgumentParser()    

    # core
    ap.add_argument("--optuna_study_name", type=str, default="study_name")
    ap.add_argument("--optuna_db", type=str, default="optuna.db")
    ap.add_argument("--imagenet_root", type=str, default="train_val")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 disables")
    ap.add_argument("--neural_ev_layer", type=str, default="encoder.layer4.0.bn1")
    ap.add_argument("--neural_data_dir", type=str, default="src/REVERSE_PRED_FINAL/majajhong_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="hpo_trials_results", help="directory to save trial results")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--lp_epochs", type=int, default=5)
    ap.add_argument("--lp_lr", type=float, default=0.1)
    ap.add_argument("--lp_wd", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--skip_knn", action="store_true")
    ap.add_argument("--skip_alpha", action="store_true")
    ap.add_argument("--skip_neural_ev", action="store_true")
    ap.add_argument("--skip_linear_probe", action="store_true")
    ap.add_argument("--skip_pr", action="store_true")

    return ap.parse_args()




args = parse_args()
study = optuna.create_study(
    storage='sqlite:///{}'.format(args.optuna_db),
    study_name=args.optuna_study_name,
    load_if_exists=True,
    direction='maximize'
)

def objective(trial):
    spectral_loss_warmup_epochs = trial.suggest_int("spectral_loss_warmup_epochs", 0, 20, step=5)
    spectral_loss_coeff = trial.suggest_float("spectral_loss_coeff", 0.0, 1.0, step=0.001)
    
    kwargs = {"imagenet_root": args.imagenet_root, "epochs": args.epochs, "batch_size": args.batch_size,
            "img_size": args.image_size, "tau": args.tau, "lr": args.lr, "wd": args.wd, "workers": args.workers, "accum_steps": args.accum_steps,
            "warmup_epochs": args.warmup_epochs, "amp": args.amp, "grad_clip": args.grad_clip, "limit_train": args.limit_train, "limit_val": args.limit_val,
            "log_every": args.log_every, "save_dir": args.save_dir, "skip_knn": args.skip_knn, "skip_alpha": args.skip_alpha, "skip_neural_ev": args.skip_neural_ev,
            "skip_linear_probe": args.skip_linear_probe, "skip_pr": args.skip_pr, "eval_every": args.eval_every, "lp_epochs": args.lp_epochs, "lp_lr": args.lp_lr,
            "lp_wd": args.lp_wd, "spectral_loss_coeff": spectral_loss_coeff, "spectral_loss_warmup_epochs": spectral_loss_warmup_epochs,
            "neural_ev_layer": args.neural_ev_layer, "neural_data_dir": args.neural_data_dir, "seed": args.seed}
    bpi = main(**kwargs)
    return bpi

study.optimize(objective, n_trials=1)