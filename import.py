import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from itertools import combinations
from itertools import islice
import random
import matplotlib.pyplot as plt
from collections import Counter
#from sklearn.cluster import KMeans
from collections import defaultdict
from itertools import product
from torchinfo import summary
from torch_geometric.data import Batch
from torchviz import make_dot
import numpy as np
import datetime
import os



# === 乱数シードを固定して再現性を担保 ===
# データのシャッフルやモデルの初期化などで乱数が使われる場合に結果を再現可能にする。
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# === ユーティリティ関数 ===

# 同型グループを解析してリスト化する関数
# ファイル形式: 各行に同型グループのノード番号が記載されている。
def parse_isomorphic_groups_multiple(file_paths):
    all_groups = []  # 全ての同型グループを格納するリスト
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    parts = line.split(': ')  # コロンで分割
                    if len(parts) == 2:  # 正しいフォーマットのみ処理
                        node_ids = list(map(int, parts[1].strip().split(', ')))  # コンマ区切りで分割して整数に変換
                        all_groups.append(node_ids)
                except ValueError:
                    print(f"Warning: Failed to parse line: {line}")

    return all_groups

# グラフデータベースを解析して辞書形式に変換する関数
# グラフのノード数、エッジ、特徴量が含まれるファイルを解析する。
def parse_graph_data(file_path):
    """
    グラフデータファイルを解析し、グラフオブジェクトを作成
    """
    graphs = {}
    with open(file_path, 'r') as file:
        for line in file.readlines()[1:]:  # ヘッダー行をスキップ
            parts = line.split('#')[0].strip().split('\t')
            node_id = int(parts[0])  # ノードID
            num_nodes = int(parts[1])  # ノード数
            # エッジリストを解析
            edge_list = [list(map(int, edge.split('-'))) for edge in parts[3].split(', ')]
            edges = [list(edge) for edge in zip(*edge_list)]
            # 特徴量を解析
            features = parse_features(parts[4], num_nodes)  # num_nodes を渡して特徴量を取得

            # グラフオブジェクトを作成して辞書に格納
            graphs[node_id] = create_graph(num_nodes, edges, features, node_id)
    return graphs



# ノード特徴量の解析
def parse_features(raw_features, num_nodes):
    """
    特徴量データを解析し、指定されたノード数分の特徴量を取得
    """
    raw_features = raw_features.strip()  # 不要な空白を削除

    # 特徴量部分だけを抽出
    features = []
    for segment in raw_features.split():
        if ':' in segment:  # "0: ..." の形式から ":" 以降を取得
            features.extend(segment.split(':')[1].split())  # ":"以降のデータを抽出
        else:
            features.append(segment)

    # 特徴量数を検証
    if len(features) != num_nodes:
        raise ValueError(f"Expected {num_nodes} features, but got {len(features)}. Features: {features}")
    
    # 特徴量をテンソルに変換
    feature_tensor = torch.tensor(list(map(float, features)), dtype=torch.float)
    return feature_tensor.view(num_nodes, -1)  # 特徴量を (num_nodes, 1) の形状に整形



# グラフオブジェクトを作成する関数
def create_graph(num_nodes, edges, features, node_id):
    edge_index = torch.tensor(edges, dtype=torch.long)  # エッジリストをテンソル化
    x = features.clone().detach().float().view(num_nodes, -1)  # 特徴量をテンソル化
    return Data(x=x, edge_index=edge_index, node_id=node_id)

def load_train_test_data(train_file_path, test_file_path):
    # 訓練データ
    train_graphs = parse_graph_data(train_file_path)
    # テストデータ
    test_graphs = parse_graph_data(test_file_path)
    return train_graphs, test_graphs


# === データ準備 ===
# グラフのペアとラベルを準備する関数
def prepare_train_pairs_with_labels(graphs, iso_groups, max_same_iso_pairs, max_non_iso_pairs, final_pairs):
    same_iso_pairs = []  # 同型ペアを格納するリスト
    for group in iso_groups:
        valid_group = [node_id for node_id in group if node_id in graphs]
        if len(valid_group) > 1:
            group_pairs = list(combinations(valid_group, 2))
            sampled_pairs = random.sample(group_pairs, min(len(group_pairs), max_same_iso_pairs - len(same_iso_pairs)))
            same_iso_pairs.extend(sampled_pairs)
        if len(same_iso_pairs) >= max_same_iso_pairs:
            break
    print("train_same")  # ログ1回だけ

    non_iso_pairs = set()  # 非同型ペアを格納する集合
    all_node_ids = list(graphs.keys())
    while len(non_iso_pairs) < max_non_iso_pairs:
        node_id1, node_id2 = random.sample(all_node_ids, 2)
        if not any(node_id1 in group and node_id2 in group for group in iso_groups):
            non_iso_pairs.add((node_id1, node_id2))
    print("train_non")  # ログ1回だけ

    # ペアとラベルを作成
    pairs = same_iso_pairs + list(non_iso_pairs)
    labels = [1.0] * len(same_iso_pairs) + [0.0] * len(non_iso_pairs)

    # ランダムに final_pairs を選択
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    selected_data = combined[:final_pairs]

    # グラフペアリストに変換
    data = []
    for (node_id1, node_id2), label in selected_data:
        graph1, graph2 = graphs[node_id1], graphs[node_id2]
        data.append((graph1, graph2, label))

    return data


def prepare_test_pairs_with_labels(graphs, iso_groups, max_same_iso_pairs, max_non_iso_pairs, max_pairs):
    """
    テストデータの同型ペアと非同型ペアを生成する。
    graphs: グラフデータの辞書。
    iso_groups: 同型グループのリスト。
    max_same_iso_pairs: 同型ペアの最大数。
    max_non_iso_pairs: 非同型ペアの最大数。
    final_pairs: 最終的に選択するペア数。
    """
    pairs, labels = [], []

    # --- 同型ペアの作成 ---
    same_iso_pairs = []
    for group in iso_groups:
        valid_group = [node_id for node_id in group if node_id in graphs]
        if len(valid_group) > 1:
            group_pairs = list(combinations(valid_group, 2))
            sampled_pairs = random.sample(group_pairs, min(len(group_pairs), max_same_iso_pairs - len(same_iso_pairs)))
            same_iso_pairs.extend(sampled_pairs)
        if len(same_iso_pairs) >= max_same_iso_pairs:
            break

    for node_id1, node_id2 in same_iso_pairs:
        pairs.append((node_id1, node_id2))
        labels.append(1.0)  # 同型ペアのラベルは1
    print("test_same")

    # --- 非同型ペアの作成 ---
    non_iso_pairs = []
    all_node_ids = list(graphs.keys())
    while len(non_iso_pairs) < max_non_iso_pairs:
        node_id1, node_id2 = random.sample(all_node_ids, 2)
        if not any(node_id1 in group and node_id2 in group for group in iso_groups):
            non_iso_pairs.append((node_id1, node_id2))

    for node_id1, node_id2 in non_iso_pairs:
        pairs.append((node_id1, node_id2))
        labels.append(0.0)  # 非同型ペアのラベルは0
    print("test_non")

    # --- ランダムにfinal_pairsを選択 ---
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    selected_data = combined[:max_pairs]

    # グラフペアリストに変換
    test_data = []
    for (node_id1, node_id2), label in selected_data:
        graph1, graph2 = graphs[node_id1], graphs[node_id2]
        test_data.append((graph1, graph2, label))

    return test_data


# === モデル定義 ===
# グラフ同型性を判定するニューラルネットワークモデル
class GraphIsomorphismModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(GraphIsomorphismModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)  # 通常の GCNConv を使用
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.conv4 = GCNConv(hidden_dim3, output_dim)
        self.fc = torch.nn.Linear(output_dim * 2, 1)

    def forward(self, graph1, graph2):
        embed1 = self.compute_embedding(graph1)
        embed2 = self.compute_embedding(graph2)
        combined = torch.cat([embed1, embed2], dim=-1)
        return self.fc(combined)

    def compute_embedding(self, graph):
        x = torch.nn.functional.leaky_relu(self.conv1(graph.x, graph.edge_index))
        x = torch.nn.functional.leaky_relu(self.conv2(x, graph.edge_index))
        x = torch.nn.functional.leaky_relu(self.conv3(x, graph.edge_index))
        x = torch.nn.functional.leaky_relu(self.conv4(x, graph.edge_index))
        return global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long))
    
# num_nodes = 10  # ノード数
# num_features = 16  # ノードの特徴量数
# edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)  # エッジリスト
# x = torch.randn((num_nodes, num_features))  # ノード特徴量
# graph1 = Data(x=x, edge_index=edge_index)  # グラフ1
# graph2 = Data(x=x, edge_index=edge_index)  # グラフ2（同じ構造を使用）

# # モデルインスタンス
# input_dim = num_features
# hidden_dim1 = 64
# hidden_dim2 = 64
# hidden_dim3 = 64
# output_dim = 32
# model = GraphIsomorphismModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)

# # 出力を計算して計算グラフを構築
# output = model(graph1, graph2)

# # torchviz を使用して計算グラフを可視化
# graph = make_dot(output, params=dict(model.named_parameters()))
# graph.format = "png"  # 図の形式を指定
# graph.render("graph_isomorphism_network") 

def extract_units_from_model(model):
    units = []
    for layer in model.children():  # モデルのレイヤーを順番に取得
        if isinstance(layer, GCNConv):
            units.append(layer.out_channels)  # GCNConvの出力次元数を追加
    return units

def plot_dual_graph_structure(model, input_dim, hidden_dims):
    def calculate_layer_positions(num_layers, units, y_offset=0, spacing=30):
        """
        各レイヤーのノード座標を計算する関数。
        spacing: ノード間の距離を調整。
        y_offset: グラフ全体の位置を上下に移動。
        """
        positions = []
        for layer_index, unit_count in enumerate(units):
            x = [layer_index] * unit_count  # X座標はレイヤーごとに固定
            y = np.linspace(-unit_count * spacing, unit_count * spacing, unit_count) + y_offset
            positions.append((x, y))
        return positions

    """
    Graph 1とGraph 2を完全に分離し、Global Poolingが2つのグラフを統合する形で表示。
    """
    units = [input_dim] + hidden_dims  # 各レイヤーのユニット数
    layer_names = ["Input Layer"] + [f"GCNConv {i+1}" for i in range(len(hidden_dims))] + ["Global Pooling", "Feature Combination", "Fully Connected"]

    fig, ax = plt.subplots(figsize=(20, 25))

    # --- Graph 1 のノード配置 ---
    graph1_positions = calculate_layer_positions(len(units), units, y_offset=-500, spacing=30)
    for i, (x, y) in enumerate(graph1_positions):
        ax.scatter(x, y, s=30, c="blue", label="Graph 1" if i == 0 else "")
        if i > 0:  # 前のレイヤーと接続
            for y1 in graph1_positions[i - 1][1]:
                for y2 in y:
                    ax.plot([x[0] - 1, x[0]], [y1, y2], c="gray", lw=1)

    # --- Graph 2 のノード配置 ---
    graph2_positions = calculate_layer_positions(len(units), units, y_offset=500, spacing=30)
    for i, (x, y) in enumerate(graph2_positions):
        ax.scatter(x, y, s=30, c="green", label="Graph 2" if i == 0 else "")
        if i > 0:  # 前のレイヤーと接続
            for y1 in graph2_positions[i - 1][1]:
                for y2 in y:
                    ax.plot([x[0] - 1, x[0]], [y1, y2], c="gray", lw=1)

    # --- Global Pooling 出力 ---
    global_pooling_x = [len(units)]
    global_pooling_y = [0]  # 中央に集約
    ax.scatter(global_pooling_x, global_pooling_y, s=200, c="purple", label="Global Pooling (Combined)")
    for y1 in graph1_positions[-1][1]:
        ax.plot([len(units) - 1, len(units)], [y1, global_pooling_y[0]], c="gray", lw=1.5)
    for y1 in graph2_positions[-1][1]:
        ax.plot([len(units) - 1, len(units)], [y1, global_pooling_y[0]], c="gray", lw=1.5)

    # --- Feature Combination と全結合層 ---
    fc_x = [len(units), len(units) + 1, len(units) + 2]
    fc_y = [0, 0, 0]
    ax.scatter(fc_x, fc_y, s=200, c="red", label="Fully Connected")
    ax.text(fc_x[0], fc_y[0] + 20, "Feature Combination", ha="center", fontsize=10, weight="bold")
    ax.text(fc_x[1], fc_y[1] + 20, "Fully Connected Layer", ha="center", fontsize=10, weight="bold")
    ax.text(fc_x[2], fc_y[2] + 20, "Output (0 or 1)", ha="center", fontsize=10, weight="bold")

    # Global Pooling 出力 → Feature Combination
    ax.plot([len(units), len(units) + 1], [0, 0], c="gray", lw=1.5)

    # Feature Combination → Fully Connected
    ax.plot([len(units) + 1, len(units) + 2], [0, 0], c="gray", lw=1.5)

    # --- グラフのデザイン設定 ---
    tick_positions = list(range(len(layer_names)))
    plt.xticks(tick_positions, layer_names, rotation=20, ha="right")
    plt.yticks([])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    



def plot_separate_metrics(train_losses_1, train_data_1, test_losses_1, test_data_1, 
                          train_accuracies_1, test_accuracies_1, lr_1, 
                          train_losses_2, train_data_2, test_losses_2, test_data_2, 
                          train_accuracies_2, test_accuracies_2, lr_2, 
                          model, epochs_1, epochs_2):
    """
    損失と正解率を2回目の学習も含めて分けてプロットする関数
    """ 
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    cwd = os.getcwd()  # 現在の作業ディレクトリ

    # 損失のプロット
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_1, train_losses_1, label="Train Loss (1st)")
    plt.plot(epochs_1, test_losses_1, label="Test Loss (1st)")
    plt.plot(epochs_2, train_losses_2, label="Train Loss (2nd)")
    plt.plot(epochs_2, test_losses_2, label="Test Loss (2nd)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")  # 対数スケールで表示
    plt.legend()
    plt.grid()

    # ハイパーパラメータ表示
    hyperparameters_1 = (
        f"1st Training:\n"
        f"  Learning Rate: {lr_1}\n"
        f"  Train Pairs: {len(train_data_1)}\n"
        f"  Test Pairs: {len(test_data_1)}"
    )
    hyperparameters_2 = (
        f"2nd Training:\n"
        f"  Learning Rate: {lr_2}\n"
        f"  Train Pairs: {len(train_data_2)}\n"
        f"  Test Pairs: {len(test_data_2)}"
    )
    plt.text(0.01, 0.95, hyperparameters_1, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round", alpha=0.2))
    plt.text(0.01, 0.60, hyperparameters_2, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round", alpha=0.2))

    filename_loss = f'{cwd}/loss_{now}.png'
    plt.savefig(filename_loss)
    print(f"Loss plot saved as {filename_loss}")
    plt.show()
    plt.close()

    # 正解率のプロット
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_1, train_accuracies_1, label="Train Accuracy (1st)", linestyle="--")
    plt.plot(epochs_1, test_accuracies_1, label="Test Accuracy (1st)")
    plt.plot(epochs_2, train_accuracies_2, label="Train Accuracy (2nd)", linestyle="--")
    plt.plot(epochs_2, test_accuracies_2, label="Test Accuracy (2nd)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    # ハイパーパラメータ再表示
    plt.text(0.01, 0.95, hyperparameters_1, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round", alpha=0.2))
    plt.text(0.01, 0.60, hyperparameters_2, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round", alpha=0.2))

    filename_accuracy = f'{cwd}/accuracy_{now}.png'
    plt.savefig(filename_accuracy)
    print(f"Accuracy plot saved as {filename_accuracy}")
    plt.show()
    plt.close()


# === 学習と評価を統合した関数 ===
def train_and_evaluate_with_loss_tracking(train_data, test_data, model, epochs, lr, seed):
    set_seed(seed)  # 乱数シードの固定
    print("test8")
    # モデルとデバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 損失と正解率の記録用リスト
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 評価関数（テストデータを使用）
    def evaluate(data):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0  # 混同行列の初期化
        results = []  # 詳細結果を記録するリスト

        criterion = torch.nn.BCEWithLogitsLoss()  # 損失関数

        with torch.no_grad():
            for graph1, graph2, label in data:
                graph1, graph2 = graph1.to(device), graph2.to(device)
                label = torch.tensor([label], dtype=torch.float, device=device)
                output = model(graph1, graph2).squeeze()
                loss = criterion(output.unsqueeze(0), label)
                total_loss += loss.item()

                prediction = torch.sigmoid(output).item()
                predicted_label = 1 if prediction >= 0.5 else 0
                correct += (predicted_label == label.item())
                total += 1

                # 混同行列の更新
                if label.item() == 1 and predicted_label == 1:
                    tp += 1
                elif label.item() == 0 and predicted_label == 0:
                    tn += 1
                elif label.item() == 0 and predicted_label == 1:
                    fp += 1
                elif label.item() == 1 and predicted_label == 0:
                    fn += 1

                # 詳細結果の記録
                results.append({
                    "graph1_id": graph1.node_id,
                    "graph2_id": graph2.node_id,
                    "prediction": prediction,
                    "label": label.item(),
                    "predicted_label": predicted_label
                })

        accuracy = (tp + tn) / total * 100
        average_loss = total_loss / len(data)

        return average_loss, accuracy, tp, fp, tn, fn, results

    # トレーニングループ
    for epoch in range(epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        for graph1, graph2, label in train_data:  # train_dataの各ペア（graph1, graph2, label）を処理
            graph1, graph2 = graph1.to(device), graph2.to(device)  # デバイスに移動
            label = torch.tensor([label], dtype=torch.float, device=device)  # ラベルもデバイスに移動
            optimizer.zero_grad()  # 勾配を初期化
            output = model(graph1, graph2).squeeze()  # モデルで予測
            
            loss = criterion(output.unsqueeze(0), label)  # 損失計算
            #print(f"Graph1 ID: {graph1.node_id}, Graph2 ID: {graph2.node_id}, Output: {output.item()}, Label: {label.item()}, Loss: {loss.item()}")
            #print(f"Model Output (Sigmoid): {torch.sigmoid(output).item()}, Loss: {loss.item()}")
            loss.backward()  # バックプロパゲーション
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} Gradient Norm: {param.grad.norm().item()}")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} Gradient Norm: {param.grad.norm()}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配のクリッピング
            optimizer.step()  # パラメータ更新

            total_train_loss += loss.item()

            # 正解率の計算
            prediction = torch.sigmoid(output).item()
            predicted_label = 1 if prediction >= 0.5 else 0
            correct_train += (predicted_label == label.item())
            total_train += 1

        train_loss = total_train_loss / len(train_data)
        train_accuracy = correct_train / total_train * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        # テストデータの評価
        test_loss, test_accuracy, tp, fp, tn, fn, test_results = evaluate(test_data)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # 10エポックごとの出力
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # 早期終了条件を無効化（コメントアウトしてテスト）
        # if test_accuracy == 100.0:
        #     break
    
    print("\nModel Output Check (Test Data):")
    for graph1, graph2, label in test_data[:5]:  # 最初の5つのテストデータを確認
        graph1, graph2 = graph1.to(device), graph2.to(device)
        with torch.no_grad():
            output = model(graph1, graph2).squeeze()
            print(f"Graph1 ID: {graph1.node_id}, Graph2 ID: {graph2.node_id}, Model Output: {output.item()}")

    test_loss, test_accuracy, tp, fp, tn, fn, test_results = evaluate(test_data)
    
    for result in test_results:
        print(f"Graph1 ID: {result['graph1_id']}, Graph2 ID: {result['graph2_id']}, "
            f"Prediction: {result['prediction']:.4f}, Label: {result['label']}, "
            f"Predicted Label: {result['predicted_label']}")

    # 最終混同行列の出力
    print("\nFinal Confusion Matrix:")
    print(f"TP: {tp}\tFP: {fp}\nFN: {fn}\tTN: {tn}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_losses, train_accuracies, test_accuracies, model   # 学習済みモデルを返す


# === ランダムにペアを選択する関数 ===
def select_random_pairs(data, num_pairs, seed):
    set_seed(seed)
    return random.sample(data, min(num_pairs, len(data)))

# === 特徴量の正規化関数 ===
# 特徴量を平均0、標準偏差1に正規化する
def normalize_features(graphs):
    for graph in graphs.values():
        graph.x = (graph.x - graph.x.mean(dim=0)) / (graph.x.std(dim=0) + 1e-9)

def init_weights(m):
    if isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            torch.nn.init.zeros_(m.lin.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def split_data(pairs, train_ratio):
    """
    ペアデータを指定された割合で訓練データとテストデータに分割。
    :param pairs: ラベル付きペアデータのリスト
    :param train_ratio: 訓練データの割合 (0.0 < train_ratio < 1.0)
    :return: 訓練データとテストデータのタプル (train_data, test_data)
    """
    random.shuffle(pairs)  # シャッフルして順序依存を防止
    train_size = int(len(pairs) * train_ratio)
    return pairs[:train_size], pairs[train_size:]

# 同型グループとグラフデータを解析

file_pairs =[
    ("1020001_L2code_to_L1num.txt","1020001_INPUT_GNN_DB.txt"),
    ("1030000_L2code_to_L1num.txt","1030000_INPUT_GNN_DB.txt")
]

test_iso_file = '1030001_L2code_to_L1num.txt'
test_graph_file = '1030001_INPUT_GNN_DB.txt'

# 訓練データ
train_all_data = []

# 各ファイルペアを処理
for iso_file, graph_file in file_pairs:
    
    # 同型グループを解析
    iso_groups = parse_isomorphic_groups_multiple([iso_file])

    # グラフデータを解析
    graphs = parse_graph_data(graph_file)
    normalize_features(graphs)

    # ペアを生成
    data = prepare_train_pairs_with_labels(graphs=graphs,iso_groups=iso_groups,max_same_iso_pairs=5000,max_non_iso_pairs=5000,final_pairs=8000)
    
    # データを統合
    train_all_data.extend(data)


train_data, test_data = split_data(train_all_data, train_ratio=0.8)  # 75% を訓練データに

# ランダムシャッフル
random.shuffle(train_data)
#random.shuffle(test_data_102)

# 訓練＆テストデータ（1030001）
test_iso_groups = parse_isomorphic_groups_multiple([test_iso_file])
test_graphs_103 = parse_graph_data(test_graph_file)
normalize_features(test_graphs_103)

# 1030001のデータを準備
test_all_data = prepare_test_pairs_with_labels(test_graphs_103, test_iso_groups, max_same_iso_pairs=2000, max_non_iso_pairs=2000, max_pairs=2000)
train_data_103, test_data_final = split_data(test_all_data, train_ratio=0.8)  # 80% を訓練データに

# ランダムシャッフル
random.shuffle(train_data_103)
random.shuffle(test_data_final)

# 訓練データとテストデータを統合
#final_train_data = train_data_102 + train_data_103  # 訓練データ合計: 12000 + 3200 = 15200
# final_test_data = test_data_102 + test_data_103_final  # テストデータ合計: 4000 + 800 = 4800

# === 修正版: モデル学習 ===
# モデル定義
input_dim = train_data[0][0].x.shape[1]
print(train_data[0][0])
#print(input_dim)
model = GraphIsomorphismModel(input_dim=input_dim, hidden_dim1=64, hidden_dim2=64, hidden_dim3=64,output_dim=32)
#model.apply(init_weights)

# plot_dual_graph_structure(
#     model=model,
#     input_dim=input_dim,
#     hidden_dims=[16, 16, 8]
# )

# example_graph = train_data_102[0][0]  # 訓練データの最初のグラフを使用
# num_nodes = example_graph.num_nodes
# num_features = example_graph.x.size(1)
# edge_index_size = example_graph.edge_index.size(1)

# # 入力データのサンプルを作成
# input_data = (
#     Data(
#         x=torch.rand(num_nodes, num_features),  # ノード特徴量
#         edge_index=torch.randint(0, num_nodes, (2, edge_index_size)),  # エッジインデックス
#         node_id=1  # 仮のノードID
#     ),
#     Data(
#         x=torch.rand(num_nodes, num_features),
#         edge_index=torch.randint(0, num_nodes, (2, edge_index_size)),
#         node_id=2
#     ),
# )

# # デバイス設定（必要に応じて CPU/GPU 切り替え）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # モデルをデバイスに送る
# model = model.to(device)

# class GNNModelWrapper(nn.Module):
#     def __init__(self, model):
#         super(GNNModelWrapper, self).__init__()
#         self.model = model

#     def forward(self, graph1_x, graph1_edge_index, graph2_x, graph2_edge_index):
#         # 受け取ったテンソルをDataオブジェクトに戻す
#         graph1 = Data(x=graph1_x, edge_index=graph1_edge_index)
#         graph2 = Data(x=graph2_x, edge_index=graph2_edge_index)
#         return self.model(graph1, graph2)

# # ラップしたモデルを作成
# wrapped_model = GNNModelWrapper(model)

# # 入力テンソルを用意
# graph1 = input_data[0]
# graph2 = input_data[1]
# input_tensors = (
#     graph1.x, graph1.edge_index,
#     graph2.x, graph2.edge_index,
# )

# # torchinfoを利用してモデル構造を表示
# summary(
#     wrapped_model,
#     input_data=input_tensors,  # モデルが期待するテンソル形式
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     row_settings=["var_names"]
# )

# モデルの学習（1回目）
# 2回目の学習を有効化するスイッチ
enable_second_training = False  # Trueで2回目の学習を実施、Falseでスキップ

# モデルの学習（1回目）
train_losses_1, test_losses_1, train_acc_1, test_acc_1, trained_model = train_and_evaluate_with_loss_tracking(
    train_data=train_all_data,
    test_data=test_all_data,
    model=model,
    epochs=200,
    lr=0.001,
    seed=42
)

if enable_second_training:
    # 1回目の一部パラメータの凍結
    for param in trained_model.conv1.parameters():
        param.requires_grad = False
    for param in trained_model.conv2.parameters():
        param.requires_grad = False
    for param in trained_model.conv3.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trained_model.parameters()), lr=0.0001, weight_decay=1e-4)

    # 2回目の学習（1030001データ）
    train_losses_2, test_losses_2, train_acc_2, test_acc_2, trained_model = train_and_evaluate_with_loss_tracking(
        train_data=train_data_103,
        test_data=test_all_data,
        model=trained_model,
        epochs=100,
        lr=0.001,
        seed=42
    )

    # 損失と正解率を統合してプロット
    epochs = list(range(1, len(train_losses_1) + len(train_losses_2) + 1))
    train_losses = train_losses_1 + train_losses_2
    test_losses = test_losses_1 + test_losses_2
    train_acc = train_acc_1 + train_acc_2
    test_acc = test_acc_1 + test_acc_2
else:
    # 1回目の結果のみでプロット
    epochs = list(range(1, len(train_losses_1) + 1))
    train_losses = train_losses_1
    test_losses = test_losses_1
    train_acc = train_acc_1
    test_acc = test_acc_1

# グラフのプロット
plot_separate_metrics(
    train_losses_1=train_losses_1,
    train_data_1=train_all_data,
    test_losses_1=test_losses_1,
    test_data_1=test_all_data,
    train_accuracies_1=train_acc_1,
    test_accuracies_1=test_acc_1,
    lr_1=0.001,  # 1回目の学習率
    train_losses_2=train_losses_2 if enable_second_training else [],
    train_data_2=train_data_103 if enable_second_training else [],
    test_losses_2=test_losses_2 if enable_second_training else [],
    test_data_2=test_data_final if enable_second_training else [],
    train_accuracies_2=train_acc_2 if enable_second_training else [],
    test_accuracies_2=test_acc_2 if enable_second_training else [],
    lr_2=0.001 if enable_second_training else None,  # 2回目の学習率
    model=trained_model,
    epochs_1=list(range(1, len(train_losses_1) + 1)),
    epochs_2=list(range(len(train_losses_1) + 1, len(train_losses_1) + len(train_losses_2) + 1)) if enable_second_training else []
)


