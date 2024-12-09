import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import config


def load_data_from_json_log(json_file):
    if json_file is None:
        return None
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_loss_epoch_curve(loss_values, experiment_name):
    sns.set(style="darkgrid")

    epochs = list(range(len(loss_values)))

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=loss_values, color='b', label='Loss', linewidth=2)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Train loss vs Epochs', fontsize=14)

    plt.legend()

    os.makedirs(os.path.join(config.PLOTS_DIR, experiment_name), exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, experiment_name, 'loss_curve.png'), format="png", dpi=300,
                bbox_inches='tight')


def plot_accuracy_epoch_curve(scores, quantity_name, experiment_name):
    sns.set(style="darkgrid")

    epochs = list(range(len(scores)))
    class_accuracy_list = []
    for item in scores:
        for key, value in item.items():
            class_accuracy_list.append(value)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=class_accuracy_list, color='b', label='Loss', linewidth=2)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(f'{quantity_name.title()}', fontsize=12)
    plt.title(f'{quantity_name.title()} vs Epochs', fontsize=14)

    plt.legend()

    os.makedirs(os.path.join(config.PLOTS_DIR, experiment_name), exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, experiment_name, f'{quantity_name}_accuracy_curve.png'), format="png",
                dpi=300,
                bbox_inches='tight')


def plot_map_epoch_curve(map_scores, quantity_name, experiment_name):
    sns.set(style="darkgrid")

    epochs = list(range(len(map_scores)))[10:]
    map_list = []
    for item in map_scores:
        for key, value in item.items():
            if value is not None:
                map_list.append(value)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=map_list, color='b', label='Loss', linewidth=2)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(f'{quantity_name.upper()}', fontsize=12)
    plt.title(f'{quantity_name.upper()} vs Epochs', fontsize=14)

    plt.legend()

    os.makedirs(os.path.join(config.PLOTS_DIR, experiment_name), exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, experiment_name, 'map_curve.png'), format="png", dpi=300,
                bbox_inches='tight')


def plot_ap_per_class(ap_per_class_scores, experiment_name):
    class_ap_dict = {}

    # get list of values for each class
    for epoch_dict in ap_per_class_scores:
        for epoch_num, class_list in epoch_dict.items():
            if class_list is None:
                continue
            for category_dict in class_list:
                for category, value in category_dict.items():
                    if category not in class_ap_dict:
                        class_ap_dict[category] = []
                    if value is None:
                        class_ap_dict[category].extend([0])
                    else:
                        if not isinstance(value, list):
                            class_ap_dict[category].extend([value * 100])
                        else:
                            class_ap_dict[category].extend([v * 100 for v in value])

    # plot
    sns.set(style="darkgrid")
    cmap = plt.colormaps['tab20']

    epochs = list(range(len(ap_per_class_scores)))[10:]
    for idx, (label, values) in enumerate(class_ap_dict.items()):
        plt.plot(epochs, values, label=label.title(), color=cmap(idx))

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel("AP Scores per category (in %)", fontsize=12)
    plt.title(f'AP Score per Category vs Epochs', fontsize=14)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    os.makedirs(os.path.join(config.PLOTS_DIR, experiment_name), exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, experiment_name, 'ap_curve.png'), format="png", dpi=300,
                bbox_inches='tight')
