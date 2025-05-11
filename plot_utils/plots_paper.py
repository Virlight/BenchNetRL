import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smoothen(data, w):
    res = np.zeros_like(data)
    for i in range(len(data)):
        if i > w:
            res[i] = np.mean(data[i - w:i])
        elif i > 0:
            res[i] = np.mean(data[:i])
        else:
            res[i] = data[i]
    return res

PRETTY_NAME_ORDER = [
    "MLP Obs. Stack 1",
    "MLP Obs. Stack 4",
    "GRU",
    "LSTM",
    "Mamba-2",
    "Mamba",
    "TrXL",
    "GTrXL"
]

PRETTY_NAME_COLORS = {
    "MLP Obs. Stack 1": "#ffbf00",
    "MLP Obs. Stack 4": "#CC6677",
    "GRU":              "#228833",
    "LSTM":             "hotpink",
    "Mamba-2":          "red",
    "Mamba":            "lightgreen",
    "TrXL":             "blue",
    "GTrXL":            "#000000",
}

plt.rcParams.update({
    "figure.figsize":   (12, 7),
    "axes.titlesize":   26,
    "axes.labelsize":   24,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  22,
})

def get_pretty_name(raw_name):
    raw_name = raw_name.lower()
    raw_name1 = raw_name
    if "gtrxl" in raw_name:
        return "GTrXL"
    if "trxl" in raw_name:
        return "TrXL"
    if "mamba2" in raw_name:
        return "Mamba-2"
    if "mamba_nobuffer" in raw_name or "mamba" in raw_name:
        return "Mamba"
    if "lstm" in raw_name:
        return "LSTM"
    if "gru" in raw_name:
        return "GRU"
    if "ppo_4" in raw_name or "260k_4" in raw_name or "1m_4" in raw_name or raw_name.endswith("_4"):
        raw_name1 = "MLP Obs. Stack 4"
    if "ppo_1" in raw_name or "260k_1" in raw_name or "1m_1" in raw_name or raw_name.endswith("_1") or "ppo_40k" in raw_name:
        raw_name1 = "MLP Obs. Stack 1"
    return raw_name1

def draw_from_combined_csv(csv_path, smoothen_w=5, title="Environment", ymin=None, save=True):
    df = pd.read_csv(csv_path)
    step_col = "global_step"

    model_raw_names = {
        col.split(" - ")[0].split(": ")[-1]: get_pretty_name(col)
        for col in df.columns 
        if "avg_episode_return" in col and "__MIN" not in col and "__MAX" not in col
    }

    models = [key for key, val in model_raw_names.items() if val in PRETTY_NAME_ORDER]
    models_sorted = sorted(models, key=lambda m: PRETTY_NAME_ORDER.index(model_raw_names[m]))

    plt.figure(figsize=(10, 6))

    for model in models_sorted:
        pretty_name = model_raw_names[model]
        prefix = f"exp_name: {model} - charts/avg_episode_return"
        mean_col = prefix
        min_col = prefix + "__MIN"
        max_col = prefix + "__MAX"

        step_vals = df[step_col].values
        mean_vals = smoothen(df[mean_col].values, smoothen_w)
        min_vals = smoothen(df[min_col].values, smoothen_w)
        max_vals = smoothen(df[max_col].values, smoothen_w)

        color = PRETTY_NAME_COLORS[pretty_name]
        plt.plot(step_vals, mean_vals,
         label=pretty_name,
         color=PRETTY_NAME_COLORS[pretty_name],
         linewidth=1.5)
        plt.fill_between(step_vals, min_vals, max_vals, alpha=0.2, color=color)

    # Axes
    plt.xlabel("Training Steps", fontsize=18)
    plt.ylabel("Average Reward", fontsize=18)
    plt.title(title, fontsize=18)

    # X-axis formatting
    min_x = 0
    max_x = int(np.max(step_vals) // 1_000_000 + 1) * 1_000_000
    plt.xlim(min_x, max_x)
    xticks = np.arange(min_x, max_x + 1, 1_000_000)
    plt.xticks(xticks, [f"{int(x / 1e6)}M" if x != 0 else "0" for x in xticks])

    # Legend
    legend = plt.legend(fontsize=14, loc="upper left")
    legend.get_frame().set_alpha(0.3)
    for line in legend.get_lines():
        line.set_linewidth(4)
    if ymin is not None:
        plt.ylim(bottom=ymin)

    plt.grid(True)
    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        save_path = f"plots/{filename}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"✅ Saved plot to {save_path}")
    #plt.show()

# === Example Use Cases ===
draw_from_combined_csv("data/memory.csv", smoothen_w=10, title="MiniGrid-Memory-S11", ymin=0.4)

draw_from_combined_csv("data/cartpole.csv", smoothen_w=10, title="CartPole-v1 Masked", ymin=0)

draw_from_combined_csv("data/lunarlander.csv", smoothen_w=10, title="LunarLander-v2 Masked", ymin=-400)

draw_from_combined_csv("data/doorkey.csv", smoothen_w=10, title="MiniGrid-Doorkey-8x8", ymin=0)

draw_from_combined_csv("data/breakout.csv", smoothen_w=10, title="Breakout-v5", ymin=0)

draw_from_combined_csv("data/pong.csv", smoothen_w=10, title="Pong-v5")

draw_from_combined_csv("data/halfcheetah.csv", smoothen_w=10, title="HalfCheetah-v4")

draw_from_combined_csv("data/walker.csv", smoothen_w=10, title="Walker2d-v4", ymin=0)

draw_from_combined_csv("data/hopper.csv", smoothen_w=10, title="Hopper-v4", ymin=0)