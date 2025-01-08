import numpy as np
import torch
import torch.nn as nn


def convert_to_sparse_tensor(dok_mtrx):
    
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor

def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K):

    user_Embedding = nn.Embedding(user_Embed_wts.size()[0], user_Embed_wts.size()[1], _weight = user_Embed_wts)
    item_Embedding = nn.Embedding(item_Embed_wts.size()[0], item_Embed_wts.size()[1], _weight = item_Embed_wts)

    test_user_ids = torch.LongTensor(test_data['user_id_2'].unique())

    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts,0, 1))

    R = sp.dok_matrix((n_users, n_items), dtype = np.float32)
    R[train_data['user_id_2'], train_data['item_id_2']] = 1.0 #交互矩阵

    R_tensor = convert_to_sparse_tensor(R)
    R_tensor_dense = R_tensor.to_dense()

    R_tensor_dense = R_tensor_dense*(-np.inf)
    R_tensor_dense = torch.nan_to_num(R_tensor_dense, nan=0.0)

    relevance_score = relevance_score+R_tensor_dense#去掉了train edge

    topk_relevance_score = torch.topk(relevance_score, K).values
    topk_relevance_indices = torch.topk(relevance_score, K).indices

    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])

    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
 
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]

    test_interacted_items = test_data.groupby('user_id_2')['item_id_2'].apply(list).reset_index()

    metrics_df = pd.merge(test_interacted_items,topk_relevance_indices_df, how= 'left', left_on = 'user_id_2',right_on = ['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df. item_id_2, metrics_df.top_rlvnt_itm)]


    metrics_df['recall'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/len(x['item_id_2']), axis = 1) 
    metrics_df['precision'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/K, axis = 1)

    def get_hit_list( item_id, top_rlvnt_itm):
        return [1 if x in set( item_id) else 0 for x in top_rlvnt_itm ]

    metrics_df['hit_list'] = metrics_df.apply(lambda x : get_hit_list(x['item_id_2'], x['top_rlvnt_itm']), axis = 1)

    def get_dcg_idcg( item_id, hit_list):
        idcg  = sum([1 / np.log1p(idx+1) for idx in range(min(len( item_id),len(hit_list)))])
        dcg =  sum([hit / np.log1p(idx+1) for idx, hit in enumerate(hit_list)])
        return dcg/idcg

    def get_cumsum(hit_list):
        return np.cumsum(hit_list)

    def get_map( item_id, hit_list, hit_list_cumsum):
        return sum([hit_cumsum*hit/(idx+1) for idx, (hit, hit_cumsum) in enumerate(zip(hit_list, hit_list_cumsum))])/len( item_id)

    metrics_df['ndcg'] = metrics_df.apply(lambda x : get_dcg_idcg(x['item_id_2'], x['hit_list']), axis = 1)
    metrics_df['hit_list_cumsum'] = metrics_df.apply(lambda x : get_cumsum(x['hit_list']), axis = 1)

    metrics_df['map'] = metrics_df.apply(lambda x : get_map(x['item_id_2'], x['hit_list'], x['hit_list_cumsum']), axis = 1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean(), metrics_df['map'].mean() 


def bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0):
  
    reg_loss = (1/2)*(userEmb0.norm().pow(2) + 
                    posEmb0.norm().pow(2)  +
                    negEmb0.norm().pow(2))/float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
        
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
    return loss, reg_loss

def bpr_loss_with_quantile(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, sample_rating,quantile_tensor):
  
    reg_loss = (1/2)*(userEmb0.norm().pow(2) + 
                    posEmb0.norm().pow(2)  +
                    negEmb0.norm().pow(2))/float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
        
    score_diff = pos_scores - neg_scores
    
    # Quantile adjustment
    weight = sample_rating - quantile_tensor
    weighted_loss = weight * torch.nn.functional.softplus(-score_diff)
    quantile_loss = torch.mean(weighted_loss)
    
        
    return quantile_loss, reg_loss

def movie_reg_loss(item_sim, batch_item_embeddings, all_item_indices, device='cpu', loss_mode='basic', margin=1.5, beta=1.0):
    """
    Compute movie regularization loss with support for multiple modes.

    Args:
        item_sim (pd.DataFrame): Precomputed similarity matrix (values in [0, 1]).
        batch_item_embeddings (torch.Tensor): Batch of item embeddings.
        all_item_indices (list): List of global item indices for the batch.
        device (str): Device to run the computation on ('cpu' or 'cuda').
        loss_mode (str): Loss mode ('basic', 'cosine', 'margin', 'dissimilar-separation').
        margin (float): Margin for dissimilar items.
        alpha (float): Weight for dissimilarity penalty.

    Returns:
        torch.Tensor: Regularization loss value.
    """

    # Move batch_item_embeddings to device
    batch_item_embeddings = batch_item_embeddings.to(device)

    # Convert similarity_matrix to tensor and move to device
    similarity_matrix = torch.tensor(item_sim.values, dtype=torch.float32).to(device)

    # Get column names of similarity_matrix
    colnames = list(item_sim.columns)

    # Find batch_indices matching global indices
    batch_indices = []
    for idx in all_item_indices:
        if idx in colnames:
            batch_indices.append(colnames.index(idx))

    # If no matching indices, return 0 loss
    if not batch_indices:
        return torch.tensor(0.0, device=device)

    batch_indices = torch.tensor(batch_indices, dtype=torch.long).to(device)

    # Get the similarity matrix for the batch
    batch_similarity_matrix = similarity_matrix[batch_indices][:, batch_indices]

    # Get upper triangular indices (excluding diagonal)
    triu_indices = torch.triu_indices(row=batch_similarity_matrix.size(0), col=batch_similarity_matrix.size(1), offset=1)

    # Extract similarity values from the upper triangular part of the matrix
    similarity_values = batch_similarity_matrix[triu_indices[0], triu_indices[1]]

    # Extract embeddings for the pairs of items
    embeddings_i = batch_item_embeddings[triu_indices[0]]
    embeddings_j = batch_item_embeddings[triu_indices[1]]

    if loss_mode == 'basic':
        # Basic similarity loss: similarity_weight * ||h_i - h_j||^2
        diff_norms = torch.norm(embeddings_i - embeddings_j, dim=1, p=2)
        basic_loss = torch.sum(similarity_values * (diff_norms ** 2))
        return basic_loss

    elif loss_mode == 'cosine':
        # Cosine similarity loss: similarity_weight * (1 - cosine_similarity)
        cosine_sim = torch.nn.functional.cosine_similarity(embeddings_i, embeddings_j, dim=1)
        cosine_loss = torch.sum(similarity_values * (1 - cosine_sim))
        return cosine_loss

    elif loss_mode == 'margin':
        # Margin-based similarity loss: (1 - similarity_weight) * max(0, margin - ||h_i - h_j||)^2
        diff_norms = torch.norm(embeddings_i - embeddings_j, dim=1, p=2)
        # print(diff_norms)
        margin_loss = torch.sum((1 - similarity_values) * torch.relu(margin - diff_norms).pow(2))
        return margin_loss
    
    elif loss_mode =='mixed-margin':
        # Margin-based similarity loss: (1 - similarity_weight) * max(0, margin - ||h_i - h_j||)^2
        diff_norms = torch.norm(embeddings_i - embeddings_j, dim=1, p=2)

        similar_loss=torch.sum(similarity_values * diff_norms)
        dissimilar_mask = (similarity_values == 0).float()
        dissimilar_loss = torch.sum(dissimilar_mask * torch.relu(margin - diff_norms))
        total_loss = similar_loss + beta * dissimilar_loss
        return total_loss

    # elif loss_mode == 'dissimilar-separation':
    #     # Weakly similar items loss
    #     diff_norms = torch.norm(embeddings_i - embeddings_j, dim=1, p=2)
    #     similar_loss = torch.sum(similarity_values * diff_norms.pow(2))

    #     # Dissimilar items loss
    #     dissimilar_mask = (similarity_values == 0).float()
    #     dissimilar_loss = torch.sum(dissimilar_mask * torch.relu(margin - diff_norms).pow(2))

    #     # Combined loss
    #     total_loss = similar_loss + alpha * dissimilar_loss
    #     return total_loss

    else:
        raise ValueError(f"Invalid loss_mode '{loss_mode}'. Choose 'basic', 'cosine', 'margin', or 'mixed-margin'.")


def user_reg_loss(user_sim,user_embeddings,device='cuda'):
    # 确保 user_embeddings 在 GPU上
    user_embeddings = user_embeddings.to(device)

    # 将 similarity_matrix 转换为 tensor 并移到 GPU
    user_sim = torch.tensor(user_sim.values, dtype=torch.float32).to(device)

    # 获取上三角索引（不包括对角线）
    triu_indices = torch.triu_indices(row=user_sim.size(0), col=user_sim.size(1), offset=1)

    # 使用索引从 user_sim 中提取上三角部分的相似度值
    similarity_values = user_sim[triu_indices[0], triu_indices[1]]

    # 使用索引从 user_embeddings 中提取对应的 embeddings
    embeddings_i = user_embeddings[triu_indices[0]]
    embeddings_j = user_embeddings[triu_indices[1]]

    # 计算 embeddings 差的 L2 范数
    diff_norms = torch.norm(embeddings_i - embeddings_j, dim=1, p=2)

    # 计算正则化损失：相似度乘以范数的平方
    regularization_loss = torch.sum(similarity_values * (diff_norms ** 2))

    # print(f"Regularization Loss: {regularization_loss.item()}")
    return regularization_loss
def evaluate_model(lightGCN, train, test, n_users, n_items, K):
    lightGCN.eval()
    with torch.no_grad():
        final_user_Embed, final_item_Embed, _, _ = lightGCN.propagate_through_layers()
        recall, precision, ndcg, _map = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train, test, K)
    return recall, precision, ndcg, _map

# Helper function to log results
def log_epoch_results(epoch, train_time, eval_time, avg_loss, mf_loss, reg_loss, recall, precision, ndcg, _map):
    print(f"Epoch: {epoch}, Train Time: {train_time:.2f}s, Eval Time: {eval_time:.2f}s, "
          f"Loss: {avg_loss:.4f}, MF Loss: {mf_loss:.4f}, Reg Loss: {reg_loss:.4f}, "
          f"Recall: {recall:.4f}, Precision: {precision:.4f}, NDCG: {ndcg:.4f}, MAP: {_map:.4f}")
    


import pandas as pd
import random

def data_loader(data, batch_size, n_usr, n_itm):
    # 获取用户与物品交互信息的 DataFrame
    interected_items_df = data.groupby('user_id_2')['item_id_2'].apply(list).reset_index()

    # 函数：从未交互的物品中采样负样本
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    # 生成用户索引列表
    indices = [x for x in range(n_usr)]

    # 随机采样用户
    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)

    users.sort()

    # 创建包含采样用户的 DataFrame
    users_df = pd.DataFrame(users, columns=['users'])

    # 只保留采样用户的交互记录
    interected_items_df = pd.merge(interected_items_df, users_df, how='right', left_on='user_id_2', right_on='users')

    # 处理可能的 NaN 值，将其替换为空列表
    interected_items_df['item_id_2'] = interected_items_df['item_id_2'].apply(lambda x: x if isinstance(x, list) else [])

    # 为每个用户采样一个正样本（已交互的物品）
    pos_items = interected_items_df['item_id_2'].apply(lambda x: random.choice(x) if x else -1).values

    # 为每个用户采样一个负样本（未交互的物品）
    neg_items = interected_items_df['item_id_2'].apply(lambda x: sample_neg(x) if x else -1).values

    # 删除无效的样本（-1）
    valid_indices = pos_items != -1
    users = [user for i, user in enumerate(users) if valid_indices[i]]
    pos_items = pos_items[valid_indices]
    neg_items = neg_items[valid_indices]

    return list(users), list(pos_items), list(neg_items)

