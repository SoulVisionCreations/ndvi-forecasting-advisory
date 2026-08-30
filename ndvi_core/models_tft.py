"""
ndvi_core.models_tft — the TFT forecaster (ProTFT_Elite) + its building blocks.

Extracted verbatim (behavior-preserving) from
Run_With_MWS_Split_Temporal_TFT_FT.py so that the training driver and the
inference service instantiate the SAME architecture and can load the same
`tft_temporal_production_ft.pt` bundle.

Architecture (Lim et al. 2021-style TFT):
  GatedLinearUnit, GatedResidualNetwork, VariableSelectionNetwork,
  InterpretableMultiHeadAttention -> ProTFT_Elite.

ProTFT_Elite.forward(x_dyn, x_stat_num, x_stat_cat, emb_idx=None)
  -> (current_ndvi + delta, attn_weights)
The Prithvi 32-d block (when prithvi_projector is set) is concatenated onto the
flat static vector before the 4 static-context GRNs + fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TFT BUILDING BLOCKS  (adapted from Lim et al. 2021 / tft_model.py reference)
# ============================================================

class GatedLinearUnit(nn.Module):
    """GLU activation: splits input in half, applies sigmoid gate to one half."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """
    GRN: core TFT block.
    LayerNorm(x + GLU(ELU(W1·x + W2·context))).
    Optional context vector conditions the transform (static enrichment).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        context_dim=0,
        dropout=0.1
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = (
            nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim > 0 else None
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        self.skip_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim else None
        )

    def forward(self, x, context=None):
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)

        return self.layer_norm(residual + hidden)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: learns per-feature importance via softmax over per-feature GRNs.
    Output = weighted sum of transformed features.
    """

    def __init__(
        self,
        input_dim,
        num_features,
        hidden_dim,
        context_dim=0,
        dropout=0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = input_dim // num_features

        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                self.feature_dim, hidden_dim, hidden_dim, dropout=dropout
            )
            for _ in range(num_features)
        ])

        self.weight_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_features,
            context_dim=context_dim, dropout=dropout
        )

    def forward(self, x, context=None):
        # Split input into per-feature chunks
        features = x.split(self.feature_dim, dim=-1)

        transformed = [grn(f) for grn, f in zip(self.feature_grns, features)]
        transformed = torch.stack(transformed, dim=-2)  # [..., num_features, hidden_dim]

        weights = self.weight_grn(x, context)            # [..., num_features]
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)

        return (transformed * weights).sum(dim=-2)       # [..., hidden_dim]


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable multi-head attention (shared value weights across heads),
    so attention weights are directly interpretable.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, T, _ = query.shape

        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)      # [B, heads, T, d_k]
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, self.d_model)
        )

        return self.W_out(attn_output), attn_weights


# ============================================================
# MODEL  (Temporal Fusion Transformer)
# ============================================================

class ProTFT_Elite(nn.Module):
    """
    TFT for NDVI forecasting.

    Drop-in replacement for ProLSTM_Elite:
      - identical __init__ signature
      - identical forward(x_dyn, x_stat_num, x_stat_cat) -> (pred, weights)
      - same residual-delta output (current_ndvi + delta)
    """

    def __init__(
        self,
        dynamic_input_size,
        num_numerical_static,
        embedding_configs,
        hidden_size=256,
        num_heads=4,
        d_model=128,
        lstm_layers=1,
        tft_dropout=0.1,
        prithvi_projector=None
    ):

        super().__init__()

        self.d_model = d_model

        # ---- Categorical embeddings (state / district / tehsil) ----
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, d)
            for n, d in embedding_configs
        ])

        total_embedding_dim = sum([d for _, d in embedding_configs])

        # Frozen-Prithvi static embedding projector (None when --use_prithvi off).
        # Its projected output joins the flat static vector, so it flows through
        # ALL 4 static context GRNs (VSN / enrichment / LSTM h0,c0) + fusion.
        self.prithvi = prithvi_projector
        prithvi_dim = prithvi_projector.out_dim if prithvi_projector is not None else 0

        # Flat static vector = numerical static + categorical embeddings + prithvi
        static_input_dim = num_numerical_static + total_embedding_dim + prithvi_dim
        self.static_input_dim = static_input_dim

        # ---- 4 static context vectors (TFT paper) ----
        self.static_context_vsn = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )
        self.static_context_enrichment = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )
        self.static_context_hidden = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )
        self.static_context_cell = GatedResidualNetwork(
            static_input_dim, d_model, d_model, dropout=tft_dropout
        )

        # ---- Temporal variable selection over dynamic features ----
        self.temporal_vsn = VariableSelectionNetwork(
            input_dim=dynamic_input_size,
            num_features=dynamic_input_size,
            hidden_dim=d_model,
            context_dim=d_model,
            dropout=tft_dropout
        )

        # ---- LSTM encoder ----
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=tft_dropout if lstm_layers > 1 else 0
        )

        self.post_lstm_glu = GatedLinearUnit(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # ---- Static enrichment ----
        self.static_enrichment = GatedResidualNetwork(
            d_model, d_model, d_model,
            context_dim=d_model,
            dropout=tft_dropout
        )

        # ---- Interpretable multi-head attention ----
        self.attention = InterpretableMultiHeadAttention(
            d_model=d_model, n_heads=num_heads, dropout=tft_dropout
        )
        self.post_attn_glu = GatedLinearUnit(d_model, d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # ---- Position-wise feedforward ----
        self.positionwise_grn = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=tft_dropout
        )

        # ---- Fusion + output head (residual delta) ----
        fusion_input_size = (
            d_model +
            static_input_dim +
            1
        )

        self.fc_init = nn.Linear(fusion_input_size, 512)

        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.GELU()
        )

        self.fc_out = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        x_dyn,
        x_stat_num,
        x_stat_cat,
        emb_idx=None
    ):

        B, T, _ = x_dyn.shape

        current_ndvi = x_dyn[:, -1, 0].unsqueeze(1)

        # ---- Static encoding ----
        embedded = [
            emb(x_stat_cat[:, i])
            for i, emb in enumerate(self.embeddings)
        ]

        full_static = torch.cat(
            [x_stat_num] + embedded,
            dim=1
        )

        # Append projected frozen-Prithvi static embedding (if enabled) so it
        # feeds the static context vectors + fusion.
        if self.prithvi is not None:
            e_proj = self.prithvi(emb_idx)
            full_static = torch.cat([full_static, e_proj], dim=1)

        c_vsn = self.static_context_vsn(full_static)          # [B, d_model]
        c_enrich = self.static_context_enrichment(full_static)
        c_h = self.static_context_hidden(full_static)
        c_c = self.static_context_cell(full_static)

        # ---- Temporal variable selection (per timestep) ----
        c_vsn_expanded = c_vsn.unsqueeze(1).expand(-1, T, -1)

        flat_input = x_dyn.reshape(B * T, -1)
        flat_context = c_vsn_expanded.reshape(B * T, -1)
        selected = self.temporal_vsn(flat_input, flat_context)
        selected = selected.view(B, T, self.d_model)

        # ---- LSTM encoder (static-initialised states) ----
        h0 = c_h.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()

        lstm_out, _ = self.lstm(selected, (h0, c0))

        gated_lstm = self.post_lstm_glu(lstm_out)
        temporal_features = self.post_lstm_norm(selected + gated_lstm)

        # ---- Static enrichment ----
        c_enrich_expanded = c_enrich.unsqueeze(1).expand(-1, T, -1)
        flat_temporal = temporal_features.reshape(B * T, -1)
        flat_enrich = c_enrich_expanded.reshape(B * T, -1)
        enriched = self.static_enrichment(flat_temporal, flat_enrich)
        enriched = enriched.view(B, T, self.d_model)

        # ---- Interpretable attention ----
        attn_out, weights = self.attention(enriched, enriched, enriched)

        gated_attn = self.post_attn_glu(attn_out)
        attn_features = self.post_attn_norm(enriched + gated_attn)

        # ---- Position-wise feedforward ----
        flat_attn = attn_features.reshape(B * T, -1)
        output_features = self.positionwise_grn(flat_attn)
        output_features = output_features.view(B, T, self.d_model)

        # Most recent timestep -> temporal representation
        context = output_features[:, -1, :]               # [B, d_model]

        # ---- Fusion + residual-delta head ----
        combined = torch.cat(
            (context, full_static, current_ndvi),
            dim=1
        )

        x = self.fc_init(combined)
        x = x + self.res_block(x)
        delta = self.fc_out(x)

        return current_ndvi + delta, weights
