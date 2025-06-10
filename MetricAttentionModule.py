import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
import time

class MetricAttentionModule(nn.Module):
    def __init__(self, input_dim, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
        super(MetricAttentionModule, self).__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)

    def attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights

    def corr1(self, X):
        B, F = X.size()
        X_ = X.unsqueeze(1)
        Q = self.Wq(X_).transpose(1, 2)
        K = self.Wk(X_).transpose(1, 2)
        V = self.Wv(X_).transpose(1, 2)
        out, weights = self.attention(Q, K, V)
        return out.squeeze(-1), weights.mean(dim=0)

    def corr2(self, X):
        mean_vecs = X.mean(dim=1, keepdim=True).repeat(1, X.size(1))
        Q = self.Wq(mean_vecs)
        K = self.Wk(mean_vecs)
        V = self.Wv(mean_vecs)
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out.squeeze(1), attn_weights.squeeze(1)

    def corr3(self, X):
        Q = self.Wq(X).transpose(0, 1)
        K = self.Wk(X).transpose(0, 1)
        V = self.Wv(X).transpose(0, 1)
        output, weights = self.attention(Q, K, V)
        return output.transpose(0, 1), weights

    def corr4(self, X):
        X_T = X.transpose(0, 1).contiguous()
        X_metrics = X_T.transpose(0, 1)
        Q = self.Wq(X_metrics).transpose(0, 1)
        K = self.Wk(X_metrics).transpose(0, 1)
        V = self.Wv(X_metrics).transpose(0, 1)
        out, weights = self.attention(Q, K, V)
        output = out.transpose(0, 1)
        return output, weights

    def forward(self, X, return_weights=False):
        out1, w1 = self.corr1(X)
        out2, w2 = self.corr2(X)
        out3, w3 = self.corr3(X)
        out4, w4 = self.corr4(X)

        fused = self.alpha * out1 + self.beta * out2 + self.gamma * out3 + self.delta * out4
        enhanced = fused * X  # Hadamard product

        if return_weights:
            total_weight = (
                self.alpha * w1 +
                self.gamma * w3 +
                self.delta * w4
            )
            return enhanced, {
                'Corr1': w1,
                'Corr2': w2,
                'Corr3': w3,
                'Corr4': w4,
                'Total': total_weight
            }
        return enhanced


def plot_attention_heatmap(attn, title, xlabel, ylabel, xticks, yticks, save_as=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn.detach().cpu().numpy(),
                cmap='viridis',
                xticklabels=xticks,
                yticklabels=yticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_top_features(attn_matrix, feature_names, top_n=10, title="Top Features by Total Attention", save_as=None):
    # fix: detach before numpy
    importance_scores = attn_matrix.sum(dim=0).cpu().detach().numpy()

    indices = importance_scores.argsort()[::-1]
    top_features = [feature_names[i] for i in indices[:top_n]]
    top_scores = importance_scores[indices[:top_n]]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_features, palette="viridis")
    plt.title(title)
    plt.xlabel("Total Attention Weight (summed across all queries)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()

if __name__ == "__main__":
    # === Step 1: Load data ===
    df = pd.read_csv("/Users/ding/Documents/Fifth_SCI/CM1.csv")
    if 'Defective' not in df.columns:
        raise ValueError("Cannot find 'Defective' column in CM1.csv")

    features = df.drop(columns=["Defective"])
    feature_names = features.columns.tolist()

    # === Step 2: Preprocess ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    sample_names = [f"Sample-{i}" for i in range(X_scaled.shape[0])]

    # === Step 3: Run through attention module on full data ===
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    sample_names = [f"Sample-{i}" for i in range(X_tensor.shape[0])]
    mam = MetricAttentionModule(input_dim=X_tensor.shape[1])
    enhanced, attn_weights = mam(X_tensor, return_weights=True)

    # === Step 4: Plot heatmaps with labels ===
    plot_attention_heatmap(attn_weights['Corr1'], "Corr1: Feature ↔ Feature (intra-instance)",
                           "Feature (Key)", "Feature (Query)", feature_names, feature_names)
    plot_attention_heatmap(attn_weights['Corr2'], "Corr2: Instance ↔ Instance",
                           "Instance (Key)", "Instance (Query)", sample_names, sample_names)
    plot_attention_heatmap(attn_weights['Corr3'], "Corr3: Feature ↔ Feature (global)",
                           "Feature (Key)", "Feature (Query)", feature_names, feature_names)
    plot_attention_heatmap(attn_weights['Corr4'], "Corr4: Feature ↔ Feature (cross)",
                           "Feature (Key)", "Feature (Query)", feature_names, feature_names)
    plot_attention_heatmap(attn_weights['Total'], "Total Attention: Weighted Feature Importance",
                           "Feature (Key)", "Feature (Query)", feature_names, feature_names)

    # === Step 5: Optional export ===
    for name, mat in attn_weights.items():
        df_mat = pd.DataFrame(mat.detach().cpu().numpy(),
                              index=feature_names if mat.shape[0] == len(feature_names) else sample_names,
                              columns=feature_names if mat.shape[1] == len(feature_names) else sample_names)
        df_mat.to_csv(f"{name}_Attention.csv")
    plot_top_features(attn_weights['Total'], feature_names, top_n=10,
                      title="Top 10 Most Influential Features (Total Attention)",
                      save_as="Top_Features_BarChart.png")


    def run_selection_experiment(X_full, y_full, feature_names, top_features, max_k=10, random_state=42):
        metrics = {
            'Top_K_Features': [],
            'F1': [],
            'Accuracy': [],
            'AUC': [],
            'MCC': [],
            'Train_Time': []
        }

        for k in range(1, max_k + 1):
            selected = top_features[:k]
            X_selected = X_full[selected].values
            y = y_full.values

            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.3, random_state=random_state, stratify=y)

            clf = RandomForestClassifier(random_state=random_state, class_weight='balanced')

            # ✅ 计时开始
            start_time = time.time()
            clf.fit(X_train, y_train)
            elapsed = time.time() - start_time

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            metrics['Top_K_Features'].append(k)
            metrics['F1'].append(f1_score(y_test, y_pred))
            metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['AUC'].append(roc_auc_score(y_test, y_prob))
            metrics['MCC'].append(matthews_corrcoef(y_test, y_pred))
            metrics['Train_Time'].append(elapsed)

        return pd.DataFrame(metrics)


    def plot_selection_trends(df_result, save_as="TopK_Selection_Trends.png"):
        plt.figure(figsize=(10, 6))
        for metric in ['F1', 'Accuracy', 'AUC', 'MCC']:
            plt.plot(df_result['Top_K_Features'], df_result[metric], marker='o', label=metric)

        plt.xlabel("Number of Top Features Used")
        plt.ylabel("Score")
        plt.title("Performance Using Top-K Attention Features")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_as)
        plt.show()
# === Step 6: Extract Top-10 Feature Names from Attention Matrix ===
    total_attn = attn_weights['Total']
    importance_scores = total_attn.sum(dim=0).cpu().detach().numpy()
    sorted_indices = importance_scores.argsort()[::-1]
    top10_features = [feature_names[i] for i in sorted_indices[:10]]

    # === Step 7: Prepare full input for ablation (not sub-sampled)
    full_features = pd.DataFrame(X_scaled, columns=feature_names)
    full_labels = df['Defective'].map({'N': 0, 'Y': 1})

    # === Step 8: Run ablation experiment
    df_ablation = run_selection_experiment(full_features, full_labels, feature_names, top10_features, max_k=30)
    df_ablation.to_csv("AblationResults.csv", index=False)
    print(df_ablation)

    # === Step 9: Plot ablation trends
    plot_selection_trends(df_ablation, save_as="Ablation_Trends.png")