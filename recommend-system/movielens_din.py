import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import os


def download_movielens_1m():
    # 下载（如果没有）
    if not os.path.exists('ml-1m'):
        os.system('wget https://files.grouplens.org/datasets/movielens/ml-1m.zip')
        os.system('unzip ml-1m.zip')

download_movielens_1m()

# ===================== 1. 配置参数 =====================
class Config:
    # 数据路径
    ratings_path = 'ml-1m/ratings.dat'
    movies_path = 'ml-1m/movies.dat'
    
    # 模型参数
    embedding_dim = 16          # 嵌入维度
    hidden_dim = 64             # 隐藏层维度
    attention_hidden_dim = 32   # 注意力层隐藏维度
    dropout_rate = 0.2          # dropout率
    max_seq_len = 20            # 用户行为序列最大长度
    
    # 训练参数
    batch_size = 256
    lr = 0.001
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(Config.seed)

# ===================== 2. 数据预处理 =====================
class MovieLensDataset(Dataset):
    def __init__(self, data, user2seq, movie_ids, max_seq_len):
        self.data = data
        self.user2seq = user2seq
        self.movie_ids = movie_ids
        self.max_seq_len = max_seq_len
        self.movie_set = set(movie_ids)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row['user_id']
        target_movie = row['movie_id']
        label = row['rating']  # 回归任务，也可转为分类
        
        # 获取用户行为序列并截断/补零
        seq = self.user2seq.get(user_id, [])
        seq = seq[-self.max_seq_len:] if len(seq) > self.max_seq_len else seq
        seq_len = len(seq)
        # 补零（0作为padding标识）
        seq = seq + [0] * (self.max_seq_len - seq_len)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'target_movie': torch.tensor(target_movie, dtype=torch.long),
            'behavior_seq': torch.tensor(seq, dtype=torch.long),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def preprocess_data():
    # 检查文件是否存在
    if not os.path.exists(Config.ratings_path):
        raise FileNotFoundError(f"ratings.dat 文件不存在，请检查路径：{Config.ratings_path}")
    if not os.path.exists(Config.movies_path):
        raise FileNotFoundError(f"movies.dat 文件不存在，请检查路径：{Config.movies_path}")
    
    # 读取数据 - 指定编码为 latin-1
    ratings = pd.read_csv(
        Config.ratings_path,
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1'
    )
    movies = pd.read_csv(
        Config.movies_path,
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    
    # 编码（确保id从1开始，0作为padding）
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings['user_id'] = user_encoder.fit_transform(ratings['user_id']) + 1
    ratings['movie_id'] = movie_encoder.fit_transform(ratings['movie_id']) + 1
    
    # 按时间排序，构建用户行为序列
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    user2seq = {}
    
    # 优化序列构建逻辑，避免空序列
    for user_id, group in ratings.groupby('user_id'):
        seq = []
        # 转换为列表操作，提升效率
        movie_list = group['movie_id'].tolist()
        # 为每个位置构建序列（当前位置之前的所有电影）
        for i in range(len(movie_list)):
            user2seq[(user_id, movie_list[i])] = movie_list[:i].copy()
    
    # 重新构造数据，关联用户-电影对应的序列
    def get_seq(row):
        return user2seq.get((row['user_id'], row['movie_id']), [])
    
    ratings['seq'] = ratings.apply(get_seq, axis=1)
    
    # 过滤掉序列为空的数据（可选，避免无行为序列的样本）
    ratings = ratings[ratings['seq'].apply(len) > 0].reset_index(drop=True)
    
    # 重建user2seq为{user_id: 所有历史序列}（适配Dataset）
    user2seq_final = {}
    for user_id, group in ratings.groupby('user_id'):
        # 取该用户最后一次行为的完整序列作为代表
        user2seq_final[user_id] = group['seq'].tolist()[-1]
    
    # 划分训练/测试集
    train_data = ratings.sample(frac=0.8, random_state=Config.seed)
    test_data = ratings.drop(train_data.index)
    
    # 构建数据集
    movie_ids = ratings['movie_id'].unique()
    train_dataset = MovieLensDataset(train_data, user2seq_final, movie_ids, Config.max_seq_len)
    test_dataset = MovieLensDataset(test_data, user2seq_final, movie_ids, Config.max_seq_len)
    
    # 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 统计维度
    num_users = len(user_encoder.classes_) + 1  # +1 for padding
    num_movies = len(movie_encoder.classes_) + 1
    
    return train_loader, test_loader, num_users, num_movies

# ===================== 3. DIN模型实现 =====================
class AttentionLayer(nn.Module):
    """DIN核心注意力层"""
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, behavior_emb, target_emb, seq_len):
        """
        Args:
            behavior_emb: [batch_size, max_seq_len, embedding_dim] 行为序列嵌入
            target_emb: [batch_size, 1, embedding_dim] 目标物品嵌入
            seq_len: [batch_size] 真实序列长度
        Returns:
            weighted_emb: [batch_size, embedding_dim] 加权后的兴趣嵌入
        """
        batch_size, max_seq_len, _ = behavior_emb.shape
        
        # 扩展target_emb到序列长度维度
        target_emb_expand = target_emb.expand(-1, max_seq_len, -1)
        
        # 拼接特征：行为emb + 目标emb + 行为emb*目标emb + 行为emb-目标emb
        concat_feat = torch.cat([
            behavior_emb,
            target_emb_expand,
            behavior_emb * target_emb_expand,
            behavior_emb - target_emb_expand
        ], dim=-1)  # [batch_size, max_seq_len, 4*embedding_dim]
        
        # 计算注意力权重
        att_weight = self.attention_net(concat_feat)  # [batch_size, max_seq_len, 1]
        att_weight = att_weight.squeeze(-1)  # [batch_size, max_seq_len]
        
        # 构建mask（padding部分权重为0）
        mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(Config.device)
        mask = mask < seq_len.unsqueeze(1)  # [batch_size, max_seq_len]
        att_weight = att_weight.masked_fill(~mask, -1e9)  # padding部分设为极小值
        att_weight = torch.softmax(att_weight, dim=-1)  # [batch_size, max_seq_len]
        
        # 加权求和
        weighted_emb = torch.bmm(att_weight.unsqueeze(1), behavior_emb).squeeze(1)  # [batch_size, embedding_dim]
        
        return weighted_emb

class DIN(nn.Module):
    def __init__(self, num_users, num_movies, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim, padding_idx=0)
        self.movie_embedding = nn.Embedding(num_movies, config.embedding_dim, padding_idx=0)
        
        # 注意力层
        self.attention_layer = AttentionLayer(
            config.embedding_dim,
            config.attention_hidden_dim,
            config.dropout_rate
        )
        
        # 预测网络
        self.predict_net = nn.Sequential(
            nn.Linear(3 * config.embedding_dim, config.hidden_dim),  # user + target + weighted_seq
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, user_id, target_movie, behavior_seq, seq_len):
        # 获取嵌入
        user_emb = self.user_embedding(user_id)  # [batch_size, embedding_dim]
        target_emb = self.movie_embedding(target_movie)  # [batch_size, embedding_dim]
        behavior_emb = self.movie_embedding(behavior_seq)  # [batch_size, max_seq_len, embedding_dim]
        
        # 计算注意力加权的兴趣嵌入
        interest_emb = self.attention_layer(
            behavior_emb,
            target_emb.unsqueeze(1),
            seq_len
        )  # [batch_size, embedding_dim]
        
        # 拼接特征
        concat_feat = torch.cat([
            user_emb,
            target_emb,
            interest_emb
        ], dim=-1)  # [batch_size, 3*embedding_dim]
        
        # 预测评分
        pred = self.predict_net(concat_feat).squeeze(-1)  # [batch_size]
        
        return pred

# ===================== 4. 训练和评估函数 =====================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/Training')
    
    for batch in pbar:
        # 数据移到设备
        user_id = batch['user_id'].to(Config.device)
        target_movie = batch['target_movie'].to(Config.device)
        behavior_seq = batch['behavior_seq'].to(Config.device)
        seq_len = batch['seq_len'].to(Config.device)
        label = batch['label'].to(Config.device)
        
        # 获取当前批次的样本数
        batch_samples = label.size(0)
        
        # 前向传播
        pred = model(user_id, target_movie, behavior_seq, seq_len)
        loss = criterion(pred, label)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失（修复：使用当前批次的实际样本数）
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        avg_batch_loss = total_loss / total_samples
        pbar.set_postfix({'avg_loss': f'{avg_batch_loss:.4f}'})
    
    # 计算整个epoch的平均损失
    avg_loss = total_loss / total_samples
    return avg_loss

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        user_id = batch['user_id'].to(Config.device)
        target_movie = batch['target_movie'].to(Config.device)
        behavior_seq = batch['behavior_seq'].to(Config.device)
        seq_len = batch['seq_len'].to(Config.device)
        label = batch['label'].to(Config.device)
        
        # 获取当前批次的样本数
        batch_samples = label.size(0)
        
        pred = model(user_id, target_movie, behavior_seq, seq_len)
        loss = criterion(pred, label)
        
        # 统计损失（修复：使用当前批次的实际样本数）
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    # 计算MAE（平均绝对误差）
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    return avg_loss, mae

# ===================== 5. 主函数 =====================
def main():
    # 数据预处理
    print("开始预处理数据...")
    try:
        train_loader, test_loader, num_users, num_movies = preprocess_data()
        print(f"数据预处理完成，用户数：{num_users}，电影数：{num_movies}")
    except Exception as e:
        print(f"数据预处理失败：{str(e)}")
        return
    
    # 初始化模型、损失函数、优化器
    model = DIN(num_users, num_movies, Config).to(Config.device)
    criterion = nn.MSELoss()  # 回归任务用MSE
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    
    # 训练循环
    best_mae = float('inf')
    print(f"开始训练，使用设备：{Config.device}")
    try:
        for epoch in range(Config.epochs):
            # 训练
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
            # 评估
            test_loss, test_mae = evaluate(model, test_loader, criterion)
            # 学习率调度
            scheduler.step()
            
            # 保存最优模型
            if test_mae < best_mae:
                best_mae = test_mae
                torch.save(model.state_dict(), 'best_din_model.pth')
            
            # 打印日志
            print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} | Best MAE: {best_mae:.4f}')
    except Exception as e:
        print(f"训练过程出错：{str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的错误栈
        return
    
    print("训练完成！最优模型已保存为 best_din_model.pth")

if __name__ == '__main__':
    main()