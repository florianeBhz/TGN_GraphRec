import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer
from model.temporal_social_attention import TemporalSocialAttentionLayer
from utils.utils import MergeLayer


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, labels, memory, neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.social_neighbor_finder = social_neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device
    self.labels=labels

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    pass

  def compute_embedding(self, memory, source_nodes, timestamps, sources_nodes_left_idx, sources_nodes_right_idx, social_timestamps,  \
    n_layers, n_neighbors=20, time_diffs=None, social_time_diffs=None,use_time_proj=True):
    pass


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder,social_neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings



class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, labels,memory, neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, labels, memory,
                                         neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device

  

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_features,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding



  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return None


  def compute_final_embedding(self, memory, source_nodes, timestamps, sources_nodes_left_idx,  \
    sources_nodes_right_idx, social_timestamps,n_layers, n_neighbors=20, time_diffs=None, \
      social_time_diffs=None,use_time_proj=True):

      pass





class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, labels, memory, neighbor_finder, social_neighbor_finder,time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):

            
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features,labels, memory,
                                                  neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding



################################################################################################################################################


class GraphWithSocialAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, labels,memory, neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphWithSocialAttentionEmbedding, self).__init__(node_features, edge_features,labels, memory,
                                                  neighbor_finder, social_neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models_user_and_items = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])


    #attention model for social network
    self.attention_models_social = torch.nn.ModuleList([TemporalSocialAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models_user_and_items[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


  def aggregate_social(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, mask):
    attention_model = self.attention_models_social[n_layer - 1]

    user_embedding_interm, item_pos_neg_embedding = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          mask)

    
    source_embedding = torch.cat([user_embedding_interm, item_pos_neg_embedding], dim=2)
    return source_embedding

#################################################################################################################################################""

  def compute_item_space_embedding(self, memory, nodes, timestamps, labels,n_layers, n_neighbors=20, time_diffs=None, \
      edge_features_embeddings=True,use_time_proj=True):
      

    nodes_torch = torch.from_numpy(nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    node_features = self.node_features[nodes_torch, :] # h = v

    if self.use_memory:
      node_features = memory[nodes, :] + node_features # h = s + v

    if n_layers == 0:
      return node_features
    else:
      
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      
      batch_labels = self.labels[edge_idxs]
      edge_features = self.edge_features[batch_labels, :] if not edge_features_embeddings else self.edge_features.weight[batch_labels]

      mask = neighbors_torch == 0

      nodes_embeddings_I = self.aggregate(n_layers, node_features,
                                        nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)


      return nodes_embeddings_I,nodes_time_embedding

      
    #return users_embeddings, source_embedding[n_users:2*n_users, :]




  ##############################################################################################################################################""

  def compute_final_embedding(self, memory, source_nodes, timestamps, labels, sources_nodes_left_idx,  \
    sources_nodes_right_idx, social_timestamps,n_layers, n_neighbors=20, time_diffs=None, \
      social_time_diffs=None,edge_features_embeddings=True,use_time_proj=True):
      

    """Recursive implementation of curr_layers temporal graph attention layers.
    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_embedding, source_nodes_time_embedding = self.compute_item_space_embedding(memory, source_nodes, timestamps, labels,n_layers, n_neighbors=20, time_diffs=None, \
      edge_features_embeddings=True,use_time_proj=True)
    
    """
    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    
    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

  
    source_node_features = self.node_features[source_nodes_torch, :] # h = v

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features # h = s + v

    if n_layers == 0:
      return source_node_features
    else:
      
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      
      batch_labels = self.labels[edge_idxs]
      edge_features = self.edge_features[batch_labels, :] if not edge_features_embeddings else self.edge_features.weight[batch_labels]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_features,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

    """

    
    n_users = len(source_nodes) //3
    user_nodes = source_nodes[:n_users]
    users_embeddings_I = source_embedding[:n_users, :]

    ### temporal sampling on social neighbors  
    social_neighbors, edge_social_idxs, edge_social_times = self.social_neighbor_finder.get_temporal_neighbor(
    user_nodes,
    social_timestamps,
    n_neighbors=n_neighbors)

    social_neighbors_torch = torch.from_numpy(social_neighbors).long().to(self.device)

    edge_social_idxs = torch.from_numpy(edge_social_idxs).long().to(self.device)

    edge_social_deltas = social_timestamps[:, np.newaxis] - edge_social_times

    edge_social_deltas_torch = torch.from_numpy(edge_social_deltas).float().to(self.device)

    edge_social_time_embeddings = self.time_encoder(edge_social_deltas_torch)

    #sources_nodes_left_idx = torch.from_numpy(sources_nodes_left_idx).long().to(self.device)  ###
    #sources_nodes_right_idx = torch.from_numpy(sources_nodes_right_idx).long().to(self.device)
    #social_timestamps_torch = torch.unsqueeze(torch.from_numpy(social_timestamps).float().to(self.device), dim=1)

    #ho_I

    """
    users_embedding_S = []
    for user in user_nodes:
      user_social_adj_list = social_neighbors[user]

      print(np.intersect1d(user_nodes,user_social_adj_list))
      
      print(max(user_nodes),max(user_social_adj_list))

      user_social_neighbor_embeddings = self.compute_item_space_embedding(memory, user_social_adj_list, timestamps, labels,n_layers, n_neighbors=20, time_diffs=None, \
        edge_features_embeddings=True,use_time_proj=True)
      #social_neighbor_embeddings.append(user_social_neighbor_embeddings)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      user_social_neighbor_embeddings = user_social_neighbor_embeddings.view(len(sources_nodes_left_idx), effective_n_neighbors, -1)
     

      social_mask = social_neighbors_torch == 0

      #social_neighbor_features =  users_embeddings_I[social_neighbors_torch,:] #ho_I

      user_embedding_S = self.aggregate_social(n_layers,source_nodes_time_embedding[user],
                                        users_embeddings_I[user],
                                        user_social_neighbor_embeddings,
                                        edge_social_time_embeddings[user],
                                        social_mask)

      users_embedding_S.append(user_embedding_S)

    users_embedding_S = torch.from_numpy(np.array(users_embedding_S)).long().to(self.device)
    """

    
    self.merger = MergeLayer(users_embeddings_I.shape[1], users_embeddings_I.shape[1],
                                    self.n_node_features,
                                    self.n_node_features)

    print(users_embeddings_I.shape)
    users_embeddings = self.merger(users_embeddings_I,users_embeddings_I)

    return users_embeddings, source_embedding[n_users:2*n_users, :]


##############################################################################################################################################################

def get_embedding_module(module_type, node_features, edge_features, labels, memory, neighbor_finder,social_neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    labels = labels,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    social_neighbor_finder=social_neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  
  elif module_type == "graph_attention_social":
    return GraphWithSocialAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    labels = labels,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    social_neighbor_finder=social_neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


