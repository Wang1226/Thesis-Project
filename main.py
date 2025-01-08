import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
from model import LightGCN
from utils import evaluate_model, log_epoch_results,  bpr_loss,data_loader,movie_reg_loss
import pandas as pd

import world
import dataloader


config = world.config
EPOCHS = config['epochs']
BATCH_SIZE = config['bpr_batch']
DECAY = config['decay']
K = config['topks']
n_layers = config['layer']
latent_dim = config['recdim']

df=dataloader.df
lambda_1=config['lambda_1']
margin_value=config['margin_value']
beta_value=config['beta']
loss_mode=config['loss_mode']

fold_results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter grid search
# for lambda_1 in [0]:
#     for lambda_2 in [0]:
#         for quantile in [0.1,0.2,0.3,0.4,0.5]:
print(f"\nStarting training with Lambda 1: {lambda_1}, margin_value: {margin_value}, beta_value: {beta_value}")
fold_results.clear()

for fold, (train_index, test_index) in enumerate(kf.split(df)):
    print(f"\nFold {fold + 1} started")
    train, test = df.iloc[train_index], df.iloc[test_index]
    n_users, n_items = df['user_id'].nunique(), df['item_id'].nunique()

    lightGCN = LightGCN(train, n_users, n_items, n_layers, latent_dim)
    optimizer = torch.optim.Adam(lightGCN.parameters(), lr=0.005)

    best_recall, best_precision, best_ndcg = 0, 0, -1
    epochs_no_improve, patience = 0, 5

    for epoch in tqdm(range(EPOCHS), desc=f"Fold {fold + 1} Training"):
        n_batch = len(train) // BATCH_SIZE
        final_loss_list, mf_loss_list, reg_loss_list = [], [], []
        
        # Training loop
        train_start_time = time.time()
        lightGCN.train()
        for _ in range(n_batch):
            optimizer.zero_grad()
            users, pos_items, neg_items = data_loader(train, BATCH_SIZE, n_users, n_items) #这里不使用quantile
            # users, pos_items, neg_items,quantile_tensor,sample_rating = data_loader_with_quantile(train, BATCH_SIZE, n_users, n_items,quantile=quantile) #这里使用quantile做weighted

            users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, _ = lightGCN.forward(users, pos_items, neg_items)
            mf_loss, reg_loss = bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) #这是原始的bpr_loss
            # mf_loss, reg_loss = bpr_loss_with_quantile(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0,sample_rating,quantile_tensor) #这是加入quantile的bpr_loss
            reg_loss *= DECAY
            final_loss = (mf_loss + reg_loss 
                          +
                          lambda_1 * movie_reg_loss(dataloader.item_sim, posEmb0, pos_items,loss_mode,margin_value,beta_value)
                            # +
                        #   lambda_2 * user_reg_loss(intersection_sim_matrix, users_emb)
                        )

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            mf_loss_list.append(mf_loss.item())
            reg_loss_list.append(reg_loss.item())
        train_time = time.time() - train_start_time

        # Evaluation
        eval_start_time = time.time()
        recall, precision, ndcg, _map = evaluate_model(lightGCN, train, test, n_users, n_items, K)
        eval_time = time.time() - eval_start_time

        # Log results
        log_epoch_results(epoch + 1, train_time, eval_time, np.mean(final_loss_list),
                        np.mean(mf_loss_list), np.mean(reg_loss_list), recall, precision, ndcg, _map)

        # Update best metrics and check for early stopping
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(lightGCN.state_dict(), f"best_model_fold_{fold + 1}.pth")
        
        if recall > best_recall:
            best_recall, best_precision = recall, precision
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch + 1 > 50 and epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    fold_results.append((best_recall, best_precision))

# Print fold results
for i, (recall, precision) in enumerate(fold_results):
    print(f"Fold {i + 1}: Best Recall: {recall:.4f}, Best Precision: {precision:.4f}")

# Print average results across folds
recalls = [recall for recall, _ in fold_results]
precisions = [precision for _, precision in fold_results]
print(f"\n Training results with Lambda 1: {lambda_1}, margin_value: {margin_value}, beta_value: {beta_value}")
print(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
