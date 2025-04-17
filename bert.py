import copy
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optimizer
import transformers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

#########################################################################
################################ Dataset ################################
#########################################################################


class BertDataset(Dataset):
    """Dataset for Bert fine-tuning"""

    def __init__(self, data: pd.DataFrame, tokenizer: transformers.AutoTokenizer):
        super().__init__()

        self.data = []

        self.tokenizer = tokenizer
        for row in data.iterrows():
            delta = row[1]["delta"]
            tokens = tokenizer.tokenize(row[1]["headlines"])
            if len(tokens) > 510:
                tokens = tokens[:510]
            tokens_final = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens_final)
            self.data.append({"token_ids": token_ids, "delta": delta})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BertDatasetArticles(Dataset):
    """Dataset for Bert fine-tuning"""

    def __init__(self, data: pd.DataFrame, tokenizer: transformers.AutoTokenizer):
        super().__init__()

        self.data = []

        self.tokenizer = tokenizer
        for row in data.iterrows():
            delta = row[1]["delta_week"]
            title = row[1]['Article_title']
            summary = row[1]['Summary']
            if title[-1] != '.':
                title += '.'
            title_summ = title + ' ' + summary # Concat title and summary to use for prediction
            tokens = tokenizer.tokenize(title_summ)
            if len(tokens) > 510:
                tokens = tokens[:510]
            tokens_final = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens_final)
            self.data.append({"token_ids": token_ids, "delta": delta})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def basic_collate_fn(batch):
    """Collate function for setting."""

    inputs = None
    outputs = None

    input_ids = [torch.tensor(dp["token_ids"]) for dp in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = (input_ids != 0) * 1
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    outputs = torch.tensor([dp["delta"] for dp in batch])
    outputs = (outputs > 0) * 1

    return inputs, outputs


#######################################################################
################################ Model ################################
#######################################################################


class BertClassifier(nn.Module):
    """DistilBert Classifier"""

    def __init__(self, distil_bert: transformers.DistilBertModel):
        super(BertClassifier, self).__init__()
        self.distil_bert = distil_bert
        self.linear = nn.Linear(self.distil_bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bert_output = self.distil_bert(input_ids, attention_mask).last_hidden_state

        # use [CLS] token embedding
        # embedding = bert_output[:, 0, :]

        # use mean pooling of non-mask layers
        mask = attention_mask.unsqueeze(-1).expand(bert_output.size()).float()
        embedding = torch.sum(bert_output * mask, 1) / mask.sum(1)

        output = self.linear(embedding)
        output = torch.squeeze(output)
        return output


#########################################################################
################################ Training ###############################
#########################################################################


def get_loss_fn():
    return nn.BCEWithLogitsLoss()


def calculate_loss(scores, labels, loss_fn):
    return loss_fn(scores, labels)


def get_optimizer(net, lr, weight_decay):
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def get_hyper_parameters():
    hidden_dim = [600, 800]
    lr = [1e-4, 1e-3]
    weight_decay = [0, 0.001]
    return hidden_dim, lr, weight_decay


def train_model(
    net,
    trn_loader,
    val_loader,
    optim,
    num_epoch=50,
    collect_cycle=5,
    patience=5,
    device="cpu",
    verbose=True,
):
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    last_improvement = 0

    loss_fn = get_loss_fn()
    if verbose:
        print("------------------------ Start Training ------------------------")
    t_start = time.time()
    for epoch in range(num_epoch):
        # Training:
        net.train()
        for inputs, labels in trn_loader:
            num_itr += 1
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            scores = net(**inputs)
            loss = calculate_loss(scores, labels.float(), loss_fn)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print(
                "Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}".format(
                    epoch + 1, num_itr, loss.item()
                )
            )

        # Validation:
        accuracy, _, _, loss, _ = get_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
            last_improvement = 0
        else:
            last_improvement += 1
            if last_improvement > patience:
                print("Stopping early")
                break

    t_end = time.time()
    if verbose:
        print("Training lasted {0:.2f} minutes".format((t_end - t_start) / 60))
        print("------------------------ Training Done ------------------------")
    stats = {
        "train_loss": train_loss,
        "train_loss_ind": train_loss_ind,
        "val_loss": val_loss,
        "val_loss_ind": val_loss_ind,
        "accuracy": best_accuracy,
    }

    return best_model, stats


def get_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = []  # true labels
    y_pred = []  # predicted labels
    total_loss = []  # loss for each batch

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            scores = net(**inputs)
            loss = calculate_loss(scores, labels.float(), loss_fn)
            pred = (scores > 0) * 1

            total_loss.append(loss.item())
            y_true.append(labels.cpu())
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_true = y_true.to("cpu")
    y_pred = y_pred.to("cpu")
    accuracy = (y_true == y_pred).sum() / y_pred.shape[0]
    total_loss = sum(total_loss) / len(total_loss)

    tp = torch.logical_and(y_true == y_pred, y_pred == 1).sum()
    tn = torch.logical_and(y_true == y_pred, y_pred == 0).sum()
    fp = torch.logical_and(y_true != y_pred, y_pred == 1).sum()
    fn = torch.logical_and(y_true != y_pred, y_pred == 0).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (
        accuracy,
        precision,
        recall,
        total_loss,
        {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats["train_loss_ind"], stats["train_loss"], label="Training loss")
    plt.plot(stats["val_loss_ind"], stats["val_loss"], label="Validation loss")
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()
