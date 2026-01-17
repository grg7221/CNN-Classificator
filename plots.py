import matplotlib.pyplot as plt
from datasets import load_dataset
from dataset.custom_tk import Tokenizer
from pandas import read_csv, concat
import plotly.express as px

def dataset_text_length():
    tk = Tokenizer(500)
    def tokenize(batch):
        return {
            "tokens": [tk.encode(x, False) for x in batch['text']]
        }

    ds_train = load_dataset("stanfordnlp/imdb")['train']
    ds_train_tok = ds_train.map(tokenize, batched=True)

    text_len = []
    for tokens in ds_train_tok['tokens']:
        text_len.append(len(tokens))

    plt.figure(figsize=(10,6))
    plt.hist(text_len, 100)

    plt.xlabel("Length")
    plt.ylabel("Freq")

    plt.savefig(f'plots/text_lengths.png', dpi=150)

    plt.show()

def train_plots(batch_size, embed_dim, num_filters):
    df = read_csv(f'metrics/b{batch_size}_d{embed_dim}_f{num_filters}.csv')

    def line_plot(df, y_cols, title, y_label, file_name):
        d = df.melt(id_vars="epoch", value_vars=y_cols, var_name="metric", value_name="value")
        fig = px.line(d, x="epoch", y="value", color="metric", markers=True, title=title, labels={"epoch": "Epoch", "value": y_label, "metric": ""})
        fig.update_layout(template="plotly_white")
        fig.write_image(f'plots/{file_name}_b{batch_size}_d{embed_dim}_f{num_filters}.png', width=900, height=500, scale=2)
        #fig.show()

    line_plot(
        df,
        y_cols=["train_loss", "val_loss"],
        title="Loss: train vs validation",
        y_label="Cross-entropy loss",
        file_name='Loss'
    )

    line_plot(
        df,
        y_cols=["accuracy", "F1"],
        title="Accuracy and F1",
        y_label="Score",
        file_name='Acc+F1'
    )

    line_plot(
        df,
        y_cols=["precision", "recall"],
        title="Precision and Recall",
        y_label="Score",
        file_name='Prec+Rec'
    )

def plot_val_loss_curves(configs, title="Validation loss across configurations"):
    dfs = []
    for label, path in configs.items():
        df = read_csv(path)
        df = df[["epoch", "val_loss"]].copy()
        df["config"] = label
        dfs.append(df)

    data = concat(dfs, ignore_index=True)

    fig = px.line(
        data,
        x="epoch",
        y="val_loss",
        color="config",
        markers=True,
        title=title,
        labels={
            "epoch": "Epoch",
            "val_loss": "Validation loss",
            "config": "Configuration",
        },
    )

    fig.update_layout(template="plotly_white")
    fig.show()

    return fig

configs = {
    "default (bs=512, emb=128, nf=100)": "metrics/b512_d128_f100.csv",
    "batch=1024": "metrics/b1024_d128_f100.csv",
    "emb_dim=256": "metrics/b512_d256_f100.csv",
    "filters=200": "metrics/b512_d128_f200.csv",
}

fig = plot_val_loss_curves(configs)
fig.write_image("plots/val_loss_comparison.png", width=900, height=500, scale=2)