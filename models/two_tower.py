import torch
import torch.nn as nn
import torch.optim as optim
from utils.training_utils import calculate_rmse

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_user_features, num_item_features):
        super(TwoTowerModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_feature_embedding = nn.Linear(num_user_features, embedding_dim)
        self.item_feature_embedding = nn.Linear(num_item_features, embedding_dim)

    def forward(self, user_indices, item_indices, user_features, item_features):
        user_embed = self.user_embedding(user_indices) + self.user_feature_embedding(user_features)
        item_embed = self.item_embedding(item_indices) + self.item_feature_embedding(item_features)
        return torch.sum(user_embed * item_embed, dim=1)

def train_two_tower_model(train_matrix, test_matrix, user_features, item_features, num_users, num_items, log_file, eval_interval, embedding_dim=16, num_epochs=100, learning_rate=0.01):
    model = TwoTowerModel(num_users, num_items, embedding_dim, user_features.shape[1], item_features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    user_indices, item_indices = torch.nonzero(torch.tensor(train_matrix)).T
    ratings = torch.tensor(train_matrix[user_indices, item_indices], dtype=torch.float32)
    
    with open(log_file, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            predictions = model(user_indices, item_indices, torch.tensor(user_features[user_indices]), torch.tensor(item_features[item_indices]))
            loss = loss_fn(predictions, ratings)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    test_user_indices, test_item_indices = torch.nonzero(torch.tensor(test_matrix)).T
                    test_ratings = torch.tensor(test_matrix[test_user_indices, test_item_indices], dtype=torch.float32)
                    test_predictions = model(test_user_indices, test_item_indices, torch.tensor(user_features[test_user_indices]), torch.tensor(item_features[test_item_indices]))
                    test_rmse = calculate_rmse(test_predictions.numpy(), test_ratings.numpy())

                log.write(f"{epoch + 1},{loss.item():.4f},{test_rmse:.4f}\n")
                log.flush()
    
    return model
