"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        self.U_fact = ScaledEmbedding(num_users,self.embedding_dim)
        self.Q_fact = ScaledEmbedding(num_items,self.embedding_dim)
        self.A_fact = ZeroEmbedding(num_users,1)
        self.B_fact = ZeroEmbedding(num_users,1)

        assert len(layer_sizes) > 0
        self.mlp_layers = []
        for i,in_size in enumerate(layer_sizes[:-1]):
            self.mlp_layers.append(torch.nn.Linear(in_size,layer_sizes[i+1]))
            self.mlp_layers.append(torch.nn.ReLU())
        self.mlp_layers.append(torch.nn.Linear(layer_sizes[-1],1))
        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        if embedding_sharing:
            self.U_reg = self.U_fact 
            self.Q_reg = self.Q_fact
        else:
            self.U_reg = ScaledEmbedding(num_users,self.embedding_dim)
            self.Q_reg = ScaledEmbedding(num_items,self.embedding_dim)


        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        # p = torch.diagonal(torch.matmul(self.U_fact(user_ids),self.Q_fact(item_ids).T)+self.A_fact(user_ids)+self.B_fact(user_ids).T)
        p = torch.sum(self.U_fact(user_ids)*self.Q_fact(item_ids),dim=1,keepdim=True) + self.A_fact(user_ids) + self.B_fact(user_ids)


        U = self.U_reg(user_ids)
        Q = self.Q_reg(item_ids)
        UQ = U*Q
        x = torch.cat([U,Q,UQ],dim=1)
        for layer in self.mlp_layers:
            x = layer(x)

        predictions = torch.flatten(p)
        score = torch.flatten(x)

        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score