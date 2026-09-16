import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv,GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        # Older MSCRS checkpoints used relation ids 1..35 with
        # num_relations=35.  Preserve those rows and initialize only the newly
        # valid final relation when loading into the corrected 36-row table.
        comp_key = 'kg_encoder.comp'
        if comp_key in state_dict and self.kg_encoder.comp is not None:
            old_comp = state_dict[comp_key]
            new_comp = self.kg_encoder.comp.detach().cpu().clone()
            if old_comp.shape != new_comp.shape and old_comp.shape[1:] == new_comp.shape[1:]:
                rows = min(old_comp.shape[0], new_comp.shape[0])
                new_comp[:rows] = old_comp[:rows]
                state_dict[comp_key] = new_comp
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict, strict=False
        )
        print(missing_keys, unexpected_keys)


class CustomGCNConv(MessagePassing):
    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index):
        # 增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算节点度数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 消息传递（邻居信息聚合）
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 直接返回聚合后的输出
        return aggr_out


class HypergraphConv(nn.Module):
    """HGNN-style hypergraph convolution using an incidence matrix.

    H is a sparse incidence matrix of shape (num_nodes, num_hyperedges).
    We compute a normalized propagation without explicitly building a dense VxV matrix.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def _safe_inv(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x + eps).reciprocal()

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """x: (V, Din), H: sparse (V, E)"""
        if not H.is_sparse:
            raise ValueError("Hypergraph incidence H must be a sparse tensor")

        x = self.dropout(x)
        xw = x @ self.weight  # (V, Dout)

        # Degrees
        dv = torch.sparse.sum(H, dim=1).to_dense()  # (V,)
        de = torch.sparse.sum(H, dim=0).to_dense()  # (E,)

        dv_inv_sqrt = torch.sqrt(self._safe_inv(dv)).unsqueeze(1)  # (V,1)
        de_inv = self._safe_inv(de).unsqueeze(1)  # (E,1)

        # Dv^{-1/2} XW
        xv = xw * dv_inv_sqrt
        # H^T (Dv^{-1/2} XW)
        xe = torch.sparse.mm(H.transpose(0, 1), xv)  # (E, Dout)
        xe = xe * de_inv
        # H ...
        xv2 = torch.sparse.mm(H, xe)  # (V, Dout)
        out = xv2 * dv_inv_sqrt
        if self.bias is not None:
            out = out + self.bias
        return out


class MMPrompt_inspired(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None,
    ):
        super(MMPrompt_inspired, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)
        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)
        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)
        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        # 1) KG (RGCN) view
        entity_kg = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds

        # 2) Co-occurrence (LightGCN) view
        entity_c1 = self.conv_c1(node_embeds, self.edge_index_c)
        entity_c2 = self.conv_c2(entity_c1, self.edge_index_c)
        entity_c3 = self.conv_c3(entity_c2, self.edge_index_c)

        # 3) Optional hypergraph (HGNN) view
        hyper_out = None
        if self.hyper_H is not None and self.hyper_convs is not None and self.hyper_lambda > 0:
            H = self.hyper_H
            if H.device != entity_kg.device:
                H = H.to(entity_kg.device)
            x = entity_kg
            for conv in self.hyper_convs:
                x = conv(x, H)
                x = F.relu(x)
            hyper_out = x

        # 4) Fuse base entity embeddings before projection
        if hyper_out is None:
            entity_embeds = (entity_c1 + entity_c2 + entity_c3 + entity_kg) / 4
        else:
            entity_embeds = (entity_c1 + entity_c2 + entity_c3 + entity_kg + self.hyper_lambda * hyper_out) / (4 + self.hyper_lambda)

        # 5) Text/Image similarity view (movie subset), injected by index_add_
        #    NOTE: this was commented out in the original code; we enable it here.
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)

        movie_embeds_ts1 = self.conv_ts1(node_features, self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1, self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 + movie_embeds_ts2) / 2

        movie_embeds_is1 = self.conv_is1(node_features, self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1, self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1 + movie_embeds_is2) / 2

        movie_embeds_mean = (movie_embeds_mean_t + movie_embeds_mean_i) / 2

        idx_to_id_tensor = self.idx_to_id_tensor.to(entity_embeds.device)
        indices = idx_to_id_tensor[: len(movie_embeds_mean)]
        entity_embeds.index_add_(0, indices, movie_embeds_mean)

        # 6) Project to prompt hidden size
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, entity_mask=None, return_entity_table=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        loss_cl = None
        entity_table = None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_table = self.get_entity_embeds()
            entity_embeds = entity_table[entity_ids]
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size
            if entity_mask is not None:
                attn_weights = attn_weights.masked_fill(~entity_mask[:, None, :], -1e4)

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                if entity_mask is None:
                    entity_rep = entity_embeds.mean(dim=1)
                else:
                    float_mask = entity_mask.to(entity_embeds.dtype).unsqueeze(-1)
                    entity_rep = (entity_embeds * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
                token_rep = token_weights_embeds.mean(dim=1)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        if return_entity_table:
            return prompt_embeds, loss_cl, entity_table
        return prompt_embeds, loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        comp_key = 'kg_encoder.comp'
        if comp_key in state_dict and self.kg_encoder.comp is not None:
            old_comp = state_dict[comp_key]
            new_comp = self.kg_encoder.comp.detach().cpu().clone()
            if old_comp.shape != new_comp.shape and old_comp.shape[1:] == new_comp.shape[1:]:
                rows = min(old_comp.shape[0], new_comp.shape[0])
                new_comp[:rows] = old_comp[:rows]
                state_dict[comp_key] = new_comp
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(missing_keys, unexpected_keys)




class MMPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None,
        # Hypergraph view (optional): incidence matrix H (|V| x |E|) as sparse COO
        hyper_H=None,
        hyper_layers: int = 2,
        hyper_lambda: float = 0.25,
        hyper_dropout: float = 0.0,
        paper_multimodal_fusion: bool = False,
        multimodal_lambda: float = 0.5,
        inspired_legacy_fusion: bool = False,
    ):
        super(MMPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.paper_multimodal_fusion = bool(paper_multimodal_fusion)
        self.multimodal_lambda = float(multimodal_lambda)
        self.inspired_legacy_fusion = bool(inspired_legacy_fusion)
        if not 0.0 <= self.multimodal_lambda <= 1.0:
            raise ValueError("multimodal_lambda must be in [0, 1]")

        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)


        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN


        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)

        # --- Optional hypergraph encoder (HGNN-style) ---
        self.hyper_lambda = float(hyper_lambda)
        self.hyper_layers = int(hyper_layers)
        self.hyper_H = None
        self.hyper_convs = None
        if hyper_H is not None:
            if not isinstance(hyper_H, torch.Tensor) or not hyper_H.is_sparse:
                raise ValueError("hyper_H must be a sparse torch.Tensor (incidence matrix)")
            self.hyper_H = hyper_H.coalesce()
            self.hyper_convs = nn.ModuleList([
                HypergraphConv(entity_hidden_size, entity_hidden_size, dropout=hyper_dropout)
                for _ in range(self.hyper_layers)
            ])
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds_kg = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds

        # The released INSPIRED recommendation checkpoint was trained with the
        # historical MMPrompt_inspired implementation (preserved in the March
        # notebook checkpoint). Its order differs from later generic code:
        # KG -> movie text/image propagation + KG-seeded co-occurrence -> add.
        if self.inspired_legacy_fusion:
            sorted_indices = self.sorted_indices.to(entity_embeds_kg.device)
            movie_features = torch.index_select(entity_embeds_kg, 0, sorted_indices)
            movie_t1 = self.conv_ts1(movie_features, self.edge_index_t_s)
            movie_t2 = self.conv_ts2(movie_t1, self.edge_index_t_s)
            # A third layer existed as a parameter but the historical forward
            # deliberately averaged only layers one and two.
            movie_i1 = self.conv_is1(movie_features, self.edge_index_i_s)
            movie_i2 = self.conv_is2(movie_i1, self.edge_index_i_s)
            movie_multimodal = ((movie_t1 + movie_t2) / 2 + (movie_i1 + movie_i2) / 2) / 2

            entity_embeds_c1 = self.conv_c1(entity_embeds_kg, self.edge_index_c)
            entity_embeds_c2 = self.conv_c2(entity_embeds_c1, self.edge_index_c)
            entity_embeds_c3 = self.conv_c3(entity_embeds_c2, self.edge_index_c)
            entity_embeds = (
                entity_embeds_kg + entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3
            ) / 4

            hyper_out = None
            if self.hyper_H is not None and self.hyper_convs is not None and self.hyper_lambda > 0:
                H = self.hyper_H
                if H.device != entity_embeds_kg.device:
                    H = H.to(entity_embeds_kg.device)
                x = entity_embeds_kg
                for conv in self.hyper_convs:
                    x = F.relu(conv(x, H))
                hyper_out = x
            if hyper_out is not None:
                entity_embeds = (4.0 * entity_embeds + self.hyper_lambda * hyper_out) / (4.0 + self.hyper_lambda)

            indices = self.idx_to_id_tensor.to(entity_embeds.device)[:len(movie_multimodal)]
            entity_embeds = entity_embeds.index_add(0, indices, movie_multimodal)
            entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
            return self.entity_proj2(entity_embeds)

        sorted_indices = self.sorted_indices.to(entity_embeds_kg.device)
        node_features = torch.index_select(entity_embeds_kg, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        if self.paper_multimodal_fusion:
            # MSCRS Eqs. (11)-(13): average layer 0 through layer 3.
            movie_embeds_mean_t = (
                node_features + movie_embeds_ts1 + movie_embeds_ts2 + movie_embeds_ts3
            ) / 4
        else:
            movie_embeds_mean_t = (movie_embeds_ts1 + movie_embeds_ts2) / 2

        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        if self.paper_multimodal_fusion:
            movie_embeds_mean_i = (
                node_features + movie_embeds_is1 + movie_embeds_is2 + movie_embeds_is3
            ) / 4
            # MSCRS Eq. (14), lambda=0.5 for ReDial in the paper.
            movie_embeds_multimodal = (
                self.multimodal_lambda * movie_embeds_mean_t
                + (1.0 - self.multimodal_lambda) * movie_embeds_mean_i
            )
        else:
            movie_embeds_mean_i = (movie_embeds_is1 + movie_embeds_is2) / 2
            movie_embeds_multimodal = (movie_embeds_mean_t + movie_embeds_mean_i) / 2

        # The paper initializes the collaborative graph with the one-layer
        # R-GCN table N1.  Preserve the legacy path for old checkpoints.
        collaborative_input = entity_embeds_kg if self.paper_multimodal_fusion else node_embeds
        entity_embeds_c1 = self.conv_c1(collaborative_input, self.edge_index_c)
        entity_embeds_c2 = self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 = self.conv_c3(entity_embeds_c2, self.edge_index_c)

        # Hypergraph view: capture high-order (set-level) co-occurrence without clique expansion
        hyper_out = None
        if self.hyper_H is not None and self.hyper_convs is not None and self.hyper_lambda > 0:
            H = self.hyper_H
            if H.device != entity_embeds_kg.device:
                H = H.to(entity_embeds_kg.device)
            x = entity_embeds_kg
            for conv in self.hyper_convs:
                x = conv(x, H)
                x = F.relu(x)
            hyper_out = x

        collaborative_mean = (
            entity_embeds_kg + entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3
        ) / 4
        if hyper_out is not None:
            # Static-hypergraph extension while preserving the scale of the
            # four-layer MSCRS collaborative representation.
            collaborative_mean = (
                4.0 * collaborative_mean + self.hyper_lambda * hyper_out
            ) / (4.0 + self.hyper_lambda)

        if self.paper_multimodal_fusion:
            # MSCRS Eqs. (15)-(16): fuse KG/collaborative tables, then add the
            # modality-specific movie table at the common item indices.
            entity_embeds = (collaborative_mean + entity_embeds_kg) / 2
        else:
            entity_embeds = collaborative_mean
        device = movie_embeds_multimodal.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_multimodal)]
        if self.paper_multimodal_fusion:
            entity_embeds = entity_embeds.index_add(0, indices, movie_embeds_multimodal)

        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, entity_mask=None, return_entity_table=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        loss_cl = None
        entity_table = None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_table = self.get_entity_embeds()
            entity_embeds = entity_table[entity_ids]
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size
            if entity_mask is not None:
                attn_weights = attn_weights.masked_fill(~entity_mask[:, None, :], -1e4)

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds

                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)
                if entity_mask is None:
                    entity_rep = entity_embeds.mean(dim=1)
                else:
                    float_mask = entity_mask.to(entity_embeds.dtype).unsqueeze(-1)
                    entity_rep = (entity_embeds * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        if return_entity_table:
            return prompt_embeds, loss_cl, entity_table
        return prompt_embeds, loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        comp_key = 'kg_encoder.comp'
        if comp_key in state_dict and self.kg_encoder.comp is not None:
            old_comp = state_dict[comp_key]
            new_comp = self.kg_encoder.comp.detach().cpu().clone()
            if old_comp.shape != new_comp.shape and old_comp.shape[1:] == new_comp.shape[1:]:
                rows = min(old_comp.shape[0], new_comp.shape[0])
                new_comp[:rows] = old_comp[:rows]
                state_dict[comp_key] = new_comp
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(missing_keys, unexpected_keys)
