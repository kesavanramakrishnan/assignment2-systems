import pandas as pd
import argparse

def convert(input_path):
    df = pd.read_csv(input_path)

    def cell(mean, std):
        return f"{mean:.2f} $\\pm$ {std:.2f}"

    df["forward"] = df.apply(
        lambda r: cell(r.forward_mean_ms, r.forward_std_ms), axis=1
    )
    df["backward"] = df.apply(
        lambda r: cell(r.backward_mean_ms, r.backward_std_ms), axis=1
    )
    df["optim"] = df.apply(
        lambda r: cell(r.optim_mean_ms, r.optim_std_ms), axis=1
    )

    table = df[["size", "forward", "backward", "optim"]].set_index("size")

    latex = table.to_latex(
        column_format="lrrr",
        caption=(
            "Forward, backward, and optimizer-step timings "
            "(ms, mean $\\pm$ std over 10 iters, 5 warmup) on B200 in FP32."
        ),
        label="tab:bench_b",
        escape=False,
        position="H",
    )

    latex = latex.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\centering")

    print(latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    convert(args.input)