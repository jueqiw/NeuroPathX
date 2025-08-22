import os
import random
from argparse import Namespace
from typing import List, Optional, Tuple
from collections import defaultdict

import shutil
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn import metrics

from utils.process_data import cpu, cpu_t
from utils.const import RESULT_FOLDER


def add_log(
    model: str,
    y_label: Tensor,
    output: Tensor,
    writer: SummaryWriter,
    hparams: Namespace,
    fold: int,
    epoch: int,
    result_dict: dict = None,
) -> float:
    y_pred = (torch.sigmoid(output) > 0.5).detach().cpu().numpy().astype(int)
    y_label = y_label.astype(int)
    f1 = metrics.f1_score(y_label, y_pred, average="macro")
    acc = metrics.accuracy_score(y_label, y_pred)
    kapper = metrics.cohen_kappa_score(y_label, y_pred)
    # compute auc
    # sort y_label and y_pred by y_pred
    output = output.squeeze().detach().cpu().numpy()
    sorted_indices = np.argsort(output)
    sorted_y_label = y_label[sorted_indices]
    sorted_y_pred = output[sorted_indices]
    auc = metrics.roc_auc_score(sorted_y_label, sorted_y_pred)
    sensitivity = metrics.recall_score(y_label, y_pred, pos_label=1)
    specificity = metrics.recall_score(y_label, y_pred, pos_label=0)

    # if model == "test":
    #     if not hparams.not_write_tensorboard:
    #         data = {
    #             "F1": [f1],
    #             "Acc": [acc],
    #             "Kappa": [kapper],
    #             "Sensitivity": [sensitivity],
    #             "Specificity": [specificity],
    #             "AUC": [auc],
    #         }
    # print("loging test image")
    # test_metric_plot = sns.pointplot(
    #     data=pd.DataFrame(data), join=False, linestyles="none"
    # )
    # writer.add_figure("test_metric", test_metric_plot.get_figure())

    if hparams.not_write_tensorboard:
        result_dict["mode"].append(model)
        result_dict["fold"].append(fold)
        result_dict["Epoch"].append(epoch)
        result_dict["Acc"].append(acc)
        result_dict["F1"].append(f1)
        result_dict["Kappa"].append(kapper)
        result_dict["Sensitivity"].append(sensitivity)
        result_dict["Specificity"].append(specificity)
        result_dict["AUC"].append(auc)
    else:
        writer.add_scalar(f"{model}_metric/F1", f1, epoch)
        writer.add_scalar(f"{model}_metric/Acc", acc, epoch)
        writer.add_scalar(f"{model}_metric/Kappa", kapper, epoch)
        writer.add_scalar(f"{model}_metric/Sensitivity", sensitivity, epoch)
        writer.add_scalar(f"{model}_metric/Specificity", specificity, epoch)
        writer.add_scalar(f"{model}_metric/AUC", auc, epoch)

    return acc


def predict(
    net: torch.nn.Module,
    X_gene: Tensor,
    X_img: Tensor,
    y: Tensor,
    criterion_recon,
    criterion_class,
    lambda_0,
    T,
    modality: str,
    hparams: Namespace,
):
    if hparams.classification_loss_type == "BCE":
        y = y.reshape(-1, 1).float()
    if modality == "joint":
        x_g_t = X_gene.clone().detach() * torch.unsqueeze(
            net.prob[1].clone().detach() > hparams.probabilistic_mask_threshold, 0
        )
        x_i_t = X_img.clone().detach() * torch.unsqueeze(
            net.prob[0].clone().detach() > hparams.probabilistic_mask_threshold, 0
        )
    if modality == "gene":
        x_g_t = X_gene.clone().detach() * torch.unsqueeze(
            net.prob[0].clone().detach() > hparams.probabilistic_mask_threshold, 0
        )
        x_i_t = torch.tensor(0).float()  # Placeholder, no meaning
    if modality == "img":
        x_i_t = X_img.clone().detach() * torch.unsqueeze(
            net.prob[0].clone().detach() > hparams.probabilistic_mask_threshold, 0
        )
        x_g_t = torch.tensor(0).float()  # Placeholder, no meaning

    surrogate_ig, y_hat, _, latent_ls = net(x_g_t, x_i_t, T, modality)
    gene_recon_loss, image_recon_loss = 0, 0
    if modality == "joint":
        gene_recon_loss = lambda_0[0] * torch.mean(
            criterion_recon(surrogate_ig[0], X_gene)
        )
        image_recon_loss = lambda_0[1] * torch.mean(
            criterion_recon(surrogate_ig[1], X_img)
        )
    if modality == "gene":
        gene_recon_loss = lambda_0[0] * torch.mean(
            criterion_recon(surrogate_ig[0], X_gene)
        )
    if modality == "img":
        image_recon_loss = lambda_0[1] * torch.mean(
            criterion_recon(surrogate_ig[0], X_img)
        )
    class_loss = lambda_0[2] * torch.mean(criterion_class(y_hat, y))
    total_loss = gene_recon_loss + image_recon_loss + class_loss
    return (
        total_loss,
        gene_recon_loss,
        image_recon_loss,
        class_loss,
        y_hat,
        latent_ls,
    )


def log_different_t_SNE_map(
    X_test_img: Tensor,
    X_test_gene: Tensor,
    latent_ls: Tensor,
    y_true: np.ndarray,
    writer: SummaryWriter,
    hparams: Namespace,
    epoch: int,
    mode: str,
):
    # log the t-SNE plot
    if hparams.input_modality == "img" or hparams.input_modality == "joint":
        # if epoch == 0 or mode == "test":
        #     X_test_img_np = X_test_img.cpu().numpy()
        #     tsne_original = TSNE(
        #         n_components=2, random_state=42, perplexity=X_test_img_np.shape[0] // 2
        #     )
        #     X_tsne = tsne_original.fit_transform(X_test_img_np)
        #     log_t_SNE_map(
        #         X_tsne=X_tsne,
        #         y=y_true,
        #         modality=hparams.input_modality,
        #         writer=writer,
        #         mode=mode,
        #         title="original_space",
        #         epoch=epoch,
        #     )
        tsne_latent = TSNE(
            n_components=2, random_state=42, perplexity=X_test_img.shape[0] // 2
        )
        X_tsne = tsne_latent.fit_transform(latent_ls.cpu().numpy())
        log_t_SNE_map(
            X_tsne=X_tsne,
            y=y_true,
            modality=hparams.input_modality,
            writer=writer,
            mode=mode,
            title="latent_space",
            epoch=epoch,
        )
    elif hparams.input_modality == "gene":
        if epoch == 0 or mode == "test":
            X_test_gene_np = X_test_gene.cpu().numpy()
            tsne_original = TSNE(
                n_components=2, random_state=42, perplexity=X_test_gene_np.shape[0] // 2
            )
            X_tsne = tsne_original.fit_transform(X_test_gene_np)
            log_t_SNE_map(
                X_tsne=X_tsne,
                y=y_true,
                modality=hparams.input_modality,
                writer=writer,
                mode=mode,
                title="original_space",
                epoch=epoch,
            )
        tsne_latent = TSNE(
            n_components=2, random_state=42, perplexity=X_test_gene.shape[0] // 2
        )

        X_tsne = tsne_latent.fit_transform(latent_ls.cpu().numpy())
        log_t_SNE_map(
            X_tsne=X_tsne,
            y=y_true,
            modality=hparams.input_modality,
            writer=writer,
            mode=mode,
            title="latent_space",
            epoch=epoch,
        )


def log_t_SNE_map(
    X_tsne: np.ndarray,
    y: np.ndarray,
    modality: str,
    writer: SummaryWriter,
    mode: str,
    title: str,
    epoch: int,
):
    # merge the t-SNE result with the label
    df = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
    df["y"] = y
    plot = sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue="y",
        data=df,
    )
    writer.add_figure(
        f"Epoch: {epoch}, {mode} - {modality}_{title}",
        plot.get_figure(),
        epoch,
    )


def log_feature_importance_map(
    modality: str, writer: SummaryWriter, prob: Tensor, epoch: int, mode: str
):
    if modality == "joint":
        img_prob = cpu(prob[0]).data.numpy()
        plt.stem(list(range(np.shape(img_prob)[0])), img_prob, basefmt="b-")
        writer.add_figure(f"Epoch: {epoch}, {mode} - gene_prob", plt.gcf(), epoch)
        gene_prob = cpu(prob[1]).data.numpy()
        plt.stem(list(range(np.shape(gene_prob)[0])), gene_prob, basefmt="b-")
        writer.add_figure(f"Epoch: {epoch}, {mode} - img_prob", plt.gcf(), epoch)
    else:
        tmp_prob = cpu(prob[0]).data.numpy()
        plt.stem(
            list(range(np.shape(tmp_prob)[0])), tmp_prob, basefmt="b-", markerfmt="."
        )
        writer.add_figure(f"Epoch: {epoch}, {mode} - {modality}_prob", plt.gcf(), epoch)


def class_label_name(hparams: Namespace):
    if hparams.n_classes == 2:
        class_label = ["CN", "AD"]
    elif hparams.n_classes == 3:
        class_label = ["CN", "MCI", "AD"]
    elif hparams.n_classes == 4:
        class_label = ["CN", "EMCI", "LMCI", "AD"]
    return class_label


def deal_with_ordinal_label(
    y_label: Tensor, y_pred: Tensor, ordinal_thres: float, hparams: Namespace
):
    Y_tr = cpu(y_label).data.numpy().sum(axis=1)
    Y_hat_tr = (cpu(y_pred).data.numpy() > ordinal_thres).cumprod(axis=1).sum(axis=1)

    if hparams.n_classes == 2:
        y_true = Y_tr - 1
        y_pred = Y_hat_tr - 1
        y_pred[y_pred < 0] = 0
    elif hparams.n_classes == 3:
        y_true = Y_tr
        y_pred = Y_hat_tr
    return y_true, y_pred


def initialize_check_losses(check_losses, s, num, test_n):
    for i in s:
        if i == "class_pred" or i == "class_true":
            check_losses[i] = -np.ones((num, test_n))
        else:
            check_losses[i] = np.zeros((num))
    return


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_train_val_test_on_ACE(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_index: np.ndarray,
    folders: list,
    X_unaffected_sibling: pd.DataFrame,
    test_fold_id: int,
    val_fold_id: int,
):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = (
        X.iloc[folders[val_fold_id]],
        y.iloc[folders[val_fold_id]],
    )
    n_control = (y_val == 1).sum()
    X_val_control = X_unaffected_sibling.sample(n_control)

    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    X_val_asd = X_val[y_val == 1]
    X_val = pd.concat([X_val_control, X_val_asd])
    y_val = pd.concat([pd.Series([0] * n_control), pd.Series([1] * len(X_val_asd))])
    X_test, y_test = (
        X.iloc[folders[test_fold_id]],
        y.iloc[folders[test_fold_id]],
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    X_train_normalized = {}
    X_val_normalized = {}
    X_test_normalized = {}

    # # fill NaN value with mean in the training set
    # for col in X_train.columns:
    #     mean = X_train[col].mean()
    #     X_train.loc[:, col] = X_train[col].fillna(mean)
    #     X_val.loc[:, col] = X_val[col].fillna(mean)
    #     X_test.loc[:, col] = X_test[col].fillna(mean)

    for col in X_train.columns:
        mean, std = X_train[col].mean(), X_train[col].std()
        X_train_normalized[col] = (X_train[col] - mean) / std
        X_val_normalized[col] = (X_val[col] - mean) / std
        X_test_normalized[col] = (X_test[col] - mean) / std

    X_train = pd.DataFrame(X_train_normalized).values
    X_val = pd.DataFrame(X_val_normalized).values
    X_test = pd.DataFrame(X_test_normalized).values

    return X_train, X_val, X_test


def normalize_img_data(
    cpu_t_,
    X_train_img: pd.DataFrame,
    X_val_img: pd.DataFrame,
    X_test_img: pd.DataFrame,
    X_train_mean_curvature: pd.DataFrame,
    X_val_mean_curvature: pd.DataFrame,
    X_test_mean_curvature: pd.DataFrame,
    X_train_gauscurv: pd.DataFrame,
    X_val_gauscurv: pd.DataFrame,
    X_test_gauscurv: pd.DataFrame,
    X_train_thicknessstd: pd.DataFrame,
    X_val_thicknessstd: pd.DataFrame,
    X_test_thicknessstd: pd.DataFrame,
    device: torch.device,
):
    X_train_normalized_img = {}
    X_val_normalized_img = {}
    X_test_normalized_img = {}
    X_train_mean_curvature_normalized = {}
    X_val_mean_curvature_normalized = {}
    X_test_mean_curvature_normalized = {}
    X_train_gauscurv_normalized = {}
    X_val_gauscurv_normalized = {}
    X_test_gauscurv_normalized = {}
    X_train_thicknessstd_normalized = {}
    X_val_thicknessstd_normalized = {}
    X_test_thicknessstd_normalized = {}

    for col in X_train_img.columns:
        mean, std = X_train_img[col].mean(), X_train_img[col].std()
        X_train_normalized_img[col] = (X_train_img[col] - mean) / std
        X_val_normalized_img[col] = (X_val_img[col] - mean) / std
        X_test_normalized_img[col] = (X_test_img[col] - mean) / std

    for col in X_train_mean_curvature.columns:
        mean_mean_curvature, std_mean_curvature = (
            X_train_mean_curvature[col].mean(),
            X_train_mean_curvature[col].std(),
        )
        X_train_mean_curvature_normalized[col] = (
            X_train_mean_curvature[col] - mean_mean_curvature
        ) / std_mean_curvature
        X_val_mean_curvature_normalized[col] = (
            X_val_mean_curvature[col] - mean_mean_curvature
        ) / std_mean_curvature
        X_test_mean_curvature_normalized[col] = (
            X_test_mean_curvature[col] - mean_mean_curvature
        ) / std_mean_curvature

    for col in X_train_gauscurv.columns:
        mean_gauscurv, std_gauscurv = (
            X_train_gauscurv[col].mean(),
            X_train_gauscurv[col].std(),
        )
        X_train_gauscurv_normalized[col] = (
            X_train_gauscurv[col] - mean_gauscurv
        ) / std_gauscurv
        X_val_gauscurv_normalized[col] = (
            X_val_gauscurv[col] - mean_gauscurv
        ) / std_gauscurv
        X_test_gauscurv_normalized[col] = (
            X_test_gauscurv[col] - mean_gauscurv
        ) / std_gauscurv

    for col in X_train_thicknessstd.columns:
        mean_thicknessstd, std_thicknessstd = (
            X_train_thicknessstd[col].mean(),
            X_train_thicknessstd[col].std(),
        )
        X_train_thicknessstd_normalized[col] = (
            X_train_thicknessstd[col] - mean_thicknessstd
        ) / std_thicknessstd
        X_val_thicknessstd_normalized[col] = (
            X_val_thicknessstd[col] - mean_thicknessstd
        ) / std_thicknessstd
        X_test_thicknessstd_normalized[col] = (
            X_test_thicknessstd[col] - mean_thicknessstd
        ) / std_thicknessstd

    X_train_img = cpu_t_(pd.DataFrame(X_train_normalized_img).values).to(device)
    X_val_img = cpu_t_(pd.DataFrame(X_val_normalized_img).values).to(device)
    X_test_img = cpu_t_(pd.DataFrame(X_test_normalized_img).values).to(device)

    X_train_mean_curvature = cpu_t_(
        pd.DataFrame(X_train_mean_curvature_normalized).values
    ).to(device)
    X_val_mean_curvature = cpu_t_(
        pd.DataFrame(X_val_mean_curvature_normalized).values
    ).to(device)
    X_test_mean_curvature = cpu_t_(
        pd.DataFrame(X_test_mean_curvature_normalized).values
    ).to(device)

    X_train_gauscurv = cpu_t_(pd.DataFrame(X_train_gauscurv_normalized).values).to(
        device
    )
    X_val_gauscurv = cpu_t_(pd.DataFrame(X_val_gauscurv_normalized).values).to(device)
    X_test_gauscurv = cpu_t_(pd.DataFrame(X_test_gauscurv_normalized).values).to(device)

    X_train_thicknessstd = cpu_t_(
        pd.DataFrame(X_train_thicknessstd_normalized).values
    ).to(device)
    X_val_thicknessstd = cpu_t_(pd.DataFrame(X_val_thicknessstd_normalized).values).to(
        device
    )
    X_test_thicknessstd = cpu_t_(
        pd.DataFrame(X_test_thicknessstd_normalized).values
    ).to(device)

    return (
        X_train_img,
        X_val_img,
        X_test_img,
        X_train_mean_curvature,
        X_val_mean_curvature,
        X_test_mean_curvature,
        X_train_gauscurv,
        X_val_gauscurv,
        X_test_gauscurv,
        X_train_thicknessstd,
        X_val_thicknessstd,
        X_test_thicknessstd,
    )


def create_y_paired_labels(y: pd.DataFrame, hparams: Namespace):
    ys = []
    for i in range(y.shape[0]):
        if y[i][0] == y[i][1]:
            ys.append(0)
        else:
            ys.append(1)

    y = np.array(ys)
    return y


def save_result_dataframe(
    result_fold_path: Path,
    df_result: pd.DataFrame,
    hparams: Namespace,
) -> None:
    result_file = result_fold_path / f"result_n_folder_{hparams.n_folder}.csv"
    overall_result_file = RESULT_FOLDER / f"{hparams.dataset}_overall_result.csv"

    df = pd.DataFrame.from_dict(df_result)
    if result_file.exists():
        # load the existing result file
        df_previous = pd.read_csv(result_file)
        # append the new result to the existing result file
        df = pd.concat([df_previous, df], ignore_index=True)
        os.remove(result_file)
    df.to_csv(result_file, index=False)

    # check the number of fold in df
    n_fold = df["fold"].nunique()
    if n_fold == 10:
        final_overall_result = defaultdict(list)
        metrics = ["Acc", "Sensitivity", "Specificity", "AUC"]
        modes = ["test", "val"]
        for epoch in range(100, hparams.n_epochs, 100):
            final_overall_result["name"].append(hparams.experiment_name)
            final_overall_result["Epoch"].append(epoch)
            for mode in modes:
                print("mode", mode)
                for metric in metrics:
                    value = (
                        (
                            df[(df["mode"] == mode) & (df["Epoch"] == epoch)].groupby(
                                "fold"
                            )[metric]
                        )
                        .mean()
                        .mean()
                    )
                    final_overall_result[f"{mode}_{metric}"].append(value)

        overall_df = pd.DataFrame.from_dict(final_overall_result)
        if overall_result_file.exists():
            df_previous = pd.read_csv(overall_result_file)
            overall_df = pd.concat([df_previous, overall_df], ignore_index=True)
            os.remove(overall_result_file)
        overall_df.to_csv(overall_result_file, index=False)


def save_result(
    mode: str,
    net: torch.nn.Module,
    X_gene: torch.Tensor,
    X_img: torch.Tensor,
    y: torch.Tensor,
    criterion_recon: torch.nn.Module,
    criterion_class: torch.nn.Module,
    lambda_0: List,
    temperature: Tensor,
    hparams: Namespace,
    result_fold_path: Path,
    fold_id: int,
    ordinal_thres: float,
    epoch: int,
    result_dict: dict,
    writer: Optional[SummaryWriter] = None,
):
    (
        total_loss,
        gene_recon_loss,
        image_recon_loss,
        class_loss,
        y_hat,
        latent_ls,
    ) = predict(
        net,
        X_gene,
        X_img,
        y,
        criterion_recon,
        criterion_class,
        lambda_0,
        temperature,
        modality=hparams.input_modality,
        hparams=hparams,
    )

    result_fold = result_fold_path / mode
    if not os.path.exists(result_fold):
        os.makedirs(result_fold)

    y_true_path = result_fold / f"y_true_folder_{fold_id}.npy"
    if os.path.exists(y_true_path):
        print(f"Remove {y_true_path}")
        os.remove(y_true_path)

    np.save(
        y_true_path,
        cpu(y).data.numpy(),
    )

    y_hat_path = result_fold / f"y_hat_folder_{fold_id}.npy"
    if os.path.exists(y_hat_path):
        print(f"Remove {y_hat_path}")
        os.remove(y_hat_path)

    y_true = cpu(y).data.numpy()
    y_prob = torch.nn.Sigmoid()(cpu(y_hat))
    y_pred = cpu(y_prob).data.numpy() > ordinal_thres

    np.save(
        y_hat_path,
        y_prob,
    )

    print(f"{mode} y_true y_pred X_gene")
    for cur_y, cur_y_hat, X in zip(y_true, y_prob, X_gene[:5, :]):
        print(f"{cur_y} {cur_y_hat} {X}")

    if hparams.n_classes == 2 and hparams.classification_loss_type == "Ordinal":
        y_true = y_true - 1
        y_pred = y_pred - 1
        y_pred[y_pred < 0] = 0

    add_log(
        mode,
        y_true,
        y_pred,
        writer,
        hparams,
        epoch,
        result_dict=result_dict,
    )

    if not hparams.not_write_tensorboard and mode == "test":
        log_different_t_SNE_map(
            X_test_img=X_img,
            X_test_gene=X_gene,
            latent_ls=latent_ls,
            y_true=y_true,
            writer=writer,
            hparams=hparams,
            epoch=epoch,
            mode="test",
        )


def preprocess_df_ACE(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # only choose rows with modality == `joint`
    input_df = input_df[input_df["modality"] == "joint"].copy()
    pathway_data_disease_control = input_df[
        (input_df["Cohort"] == "CON") | (input_df["Cohort"] == "ASD")
    ]
    img = pathway_data_disease_control.iloc[:, 181:]
    pathway = pathway_data_disease_control.iloc[:, :181]
    label = pathway["Cohort"].replace({"CON": 0, "ASD": 1}).astype(int)

    # drop column `Gender`, `Father Site ID`, `Mother Site ID`
    img = img.drop(columns=["Gender", "Father Site ID", "Mother Site ID"])
    pathway = pathway.drop(columns=["Site ID", "New ID", "modality", "Cohort"])

    return img, pathway, label


def preprocess_df_AD(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_df = input_df.drop(
        columns=[
            "ST81CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST81SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST81TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST81TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        ]
    )

    # drop the row with index 549, 886, 914, 915, 920, 923
    input_df = input_df.drop([549, 886, 914, 915, 920, 923])
    # for each row, print out the indx that the more than 50% of the columns are space
    pathway_data_disease_control = input_df[
        (input_df["DX_bl"] == "AD") | (input_df["DX_bl"] == "CN")
    ].copy()

    # check whether "PTID.1" in the column names
    if "PTID.1" in pathway_data_disease_control.columns:
        img = pathway_data_disease_control.iloc[:, -276:]
        pathway = pathway_data_disease_control.iloc[:, :-276]

        img.drop(
            columns=[
                "PTID.1",
                "DX_bl.1",
                "AGE.1",
                "PTGENDER.1",
            ],
            inplace=True,
        )
    else:
        img = pathway_data_disease_control.iloc[:, -272:]
        pathway = pathway_data_disease_control.iloc[:, :-272]
    label = pathway["DX_bl"].replace({"CN": 0, "AD": 1}).astype(int)

    pathway.drop(
        columns=[
            "PTID",
            "DX_bl",
            "AGE",
            "PTGENDER",
        ],
        inplace=True,
    )

    return img.astype(float), pathway, label
