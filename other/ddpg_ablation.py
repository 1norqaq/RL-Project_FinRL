import argparse
import itertools
import os
import subprocess
import sys

import pandas as pd


def parse_list(text, cast):
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def run_command(cmd, cwd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run DDPG ablations with fair evaluation.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for child runs.")
    parser.add_argument("--penalties", type=str, default="0.0,0.01,0.05")
    parser.add_argument("--break-steps", type=str, default="40000,120000")
    parser.add_argument("--net-dims", type=str, default="128,64;256,256", help="Use ';' between configs.")
    parser.add_argument("--max-windows", type=int, default=20, help="Use small value for quick ablations.")
    parser.add_argument("--out-dir", type=str, default="ablation_runs")
    parser.add_argument("--train-file", type=str, default="ddpg_train.py")
    parser.add_argument("--eval-file", type=str, default="ddpg_test.py")
    args = parser.parse_args()

    penalties = parse_list(args.penalties, float)
    break_steps = parse_list(args.break_steps, int)
    net_dim_texts = [x.strip() for x in args.net_dims.split(";") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = []

    for penalty, break_step, net_dims in itertools.product(penalties, break_steps, net_dim_texts):
        tag = f"p{penalty:g}_b{break_step}_n{net_dims.replace(',', 'x')}"
        run_dir = os.path.join(args.out_dir, tag)
        os.makedirs(run_dir, exist_ok=True)

        result_csv = os.path.join(run_dir, "results.csv")
        eval_curve_csv = os.path.join(run_dir, "evaluation_curves.csv")
        eval_metrics_csv = os.path.join(run_dir, "evaluation_metrics.csv")
        eval_plot_png = os.path.join(run_dir, "evaluation_plot.png")
        ckpt_root = os.path.join(run_dir, "checkpoints")

        train_cmd = [
            args.python,
            args.train_file,
            "--penalty-coef",
            str(penalty),
            "--break-step",
            str(break_step),
            "--net-dims",
            net_dims,
            "--max-windows",
            str(args.max_windows),
            "--checkpoint-root",
            ckpt_root,
            "--output-csv",
            result_csv,
        ]
        run_command(train_cmd, cwd=os.getcwd())

        eval_cmd = [
            args.python,
            args.eval_file,
            "--result-file",
            result_csv,
            "--out-curve-csv",
            eval_curve_csv,
            "--out-metrics-csv",
            eval_metrics_csv,
            "--out-plot",
            eval_plot_png,
        ]
        run_command(eval_cmd, cwd=os.getcwd())

        metrics = pd.read_csv(eval_metrics_csv)
        ddpg_row = metrics[metrics["strategy"] == "DDPG (rolling compounded)"].iloc[0]
        summary_rows.append(
            {
                "tag": tag,
                "penalty_coef": penalty,
                "break_step": break_step,
                "net_dims": net_dims,
                "cum_return": float(ddpg_row["cumulative_return"]),
                "sharpe": float(ddpg_row["annualized_sharpe"]),
                "max_drawdown": float(ddpg_row["max_drawdown"]),
                "run_dir": run_dir,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["sharpe", "cum_return"], ascending=False)
    summary_csv = os.path.join(args.out_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SUCCESS] Ablation summary saved: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
