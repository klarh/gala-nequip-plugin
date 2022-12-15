
import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from allegro._keys import EDGE_FEATURES as EDGE_FEATURE_KEY

from geometric_algebra_attention import pytorch as gala

class UnitNorm(torch.nn.Module):
    def forward(self, x):
        norm = gala.geometric_algebra.custom_norm(x)
        return x/torch.maximum(norm, torch.tensor(1e-7, device=x.device))

@compile_mode('script')
class GAlA_Module(GraphModuleMixin, torch.nn.Module):
    _ACTIVATION_FUNCTIONS = dict(
        swish=lambda *args, **kwargs: torch.nn.SiLU(),
    )

    _NORMALIZATION_FUNCTIONS = dict(
        layer=lambda dim, *args, **kwargs: torch.nn.LayerNorm(dim),
        momentum=lambda dim, *args, momentum=.99, **kwargs: (
            gala.MomentumNormalization(dim, momentum=momentum)),
        momentum_layer=lambda *args, momentum=.99, **kwargs: (
            gala.MomentumLayerNormalization(momentum=momentum)),
        unit=lambda *args, **kwargs: UnitNorm(),
    )

    def __init__(self, *args,
                 num_types,
                 latent_dim: int = 32,
                 field: str = AtomicDataDict.EDGE_ATTRS_KEY,
                 edge_invariant_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
                 edge_covariant_field: str = AtomicDataDict.EDGE_VECTORS_KEY,
                 out_field=EDGE_FEATURE_KEY,
                 invariant_scale=.5,
                 equivariant_scale=.125,
                 rank=2,
                 merge_fun='concat',
                 join_fun='concat',
                 invariant_mode='full',
                 covariant_mode='full',
                 dilation=2,
                 dropout=None,
                 mlp_layers=1,
                 num_blocks=3,
                 block_nonlinearity=True,
                 residual=True,
                 activation='swish',
                 score_normalization='layer',
                 value_normalization='layer',
                 block_normalization='layer',
                 invariant_value_normalization='momentum',
                 equivariant_value_normalization='momentum_layer',
                 use_multivectors=True,
                 include_normalized_products=False,
                 normalize_equivariant_values=False,
                 normalization_arguments=None,
                 convex_covariants=False,
                 tied_attention=False,
                 normalize_last_block=False,
                 irreps_in=None,
                 **kwargs):
        super().__init__()

        self.num_types = num_types
        self.latent_dim = latent_dim
        self.field = field
        self.edge_invariant_field = edge_invariant_field
        self.edge_covariant_field = edge_covariant_field
        self.out_field = out_field
        self.invariant_scale = invariant_scale
        self.equivariant_scale = equivariant_scale

        self.rank = rank
        self.merge_fun = merge_fun
        self.join_fun = join_fun
        self.invariant_mode = invariant_mode
        self.covariant_mode = covariant_mode
        self.dilation = dilation
        self.dropout = dropout
        self.mlp_layers = mlp_layers
        self.num_blocks = num_blocks
        self.block_nonlinearity = block_nonlinearity
        self.residual = residual
        self.activation = activation
        self.score_normalization = score_normalization
        self.value_normalization = value_normalization
        self.block_normalization = block_normalization
        self.invariant_value_normalization = invariant_value_normalization
        self.equivariant_value_normalization = equivariant_value_normalization
        self.use_multivectors = use_multivectors
        self.include_normalized_products = include_normalized_products
        self.normalize_equivariant_values = normalize_equivariant_values
        self.normalization_kwargs = normalization_arguments or {}
        self.convex_covariants = convex_covariants
        self.tied_attention = tied_attention
        self.normalize_last_block = normalize_last_block

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[],
        )

        self.irreps_out.update(
            {
                self.out_field: o3.Irreps(
                    [(self.latent_dim, (0, 1))]
                    ),
            }
        )

        self._initialize_gala_layers()
        self.junk_layer = torch.nn.Linear(4, self.latent_dim)

    def _initialize_gala_layers(self):
        if self.use_multivectors:
            InvariantAttention = gala.MultivectorAttention
            EquivariantAttention = gala.Multivector2MultivectorAttention
            TiedAttention = gala.TiedMultivectorAttention
            equivariant_dim = 8
            self.vector_upcast = gala.Vector2Multivector()
        else:
            InvariantAttention = gala.VectorAttention
            EquivariantAttention = lambda *args, **kwargs: None
            TiedAttention = gala.TiedVectorAttention
            equivariant_dim = 3
            self.vector_upcast = lambda x: x

        self.embedding_projection = torch.nn.Linear(2*self.num_types, self.latent_dim)

        common_kwargs = dict(
            reduce=False,
            rank=self.rank,
            merge_fun=self.merge_fun,
            join_fun=self.join_fun,
            invariant_mode=self.invariant_mode,
            covariant_mode=self.covariant_mode,
            include_normalized_products=self.include_normalized_products,
        )

        self.invariant_dims = InvariantAttention.get_invariant_dims(
            rank=self.rank, invariant_mode=self.invariant_mode,
            include_normalized_products=self.include_normalized_products)

        if self.tied_attention:
            layers = []
            for _ in range(self.num_blocks):
                layers.append(TiedAttention(
                    self.latent_dim,
                    self._make_score_net(),
                    self._make_value_net(self.latent_dim),
                    self._make_scale_net(),
                    convex_covariants=self.convex_covariants,
                    **common_kwargs))
            self.tied_layers = torch.nn.ModuleList(layers)
        else:
            invar_layers = []
            eqvar_layers = []
            for _ in range(self.num_blocks):
                invar_layers.append(InvariantAttention(
                    self.latent_dim,
                    self._make_score_net(),
                    self._make_value_net(self.latent_dim),
                    **common_kwargs))
                eqvar_layers.append(EquivariantAttention(
                    self.latent_dim,
                    self._make_score_net(),
                    self._make_value_net(self.latent_dim),
                    self._make_scale_net(),
                    convex_covariants=self.convex_covariants,
                    **common_kwargs))
            self.invar_layers = torch.nn.ModuleList(invar_layers)
            self.eqvar_layers = torch.nn.ModuleList(eqvar_layers)

        if self.block_nonlinearity:
            layers = [self._make_block_invariant_net() for _ in range(self.num_blocks)]
            self.block_nonlin_layers = torch.nn.ModuleList(layers)

        block_norm_layers = []
        eqvar_norm_layers = []
        for _ in range(self.num_blocks):
            block_norm_layers.extend(self._get_normalization_layers(
                'block', self.latent_dim))
            if self.use_multivectors:
                eqvar_norm_layers.extend(self._get_normalization_layers(
                    'equivariant_value', equivariant_dim))

        input_norm_layers = []
        if self.normalize_equivariant_values:
            for _ in range(self.num_blocks + 1):
                input_norm_layers.extend(self._get_normalization_layers(
                    None, self.latent_dim, force='unit'))

        self.block_norm_layers = torch.nn.ModuleList(block_norm_layers)
        self.eqvar_norm_layers = torch.nn.ModuleList(eqvar_norm_layers)
        self.input_norm_layers = torch.nn.ModuleList(input_norm_layers)

        # final_kwargs = dict(common_kwargs)
        # final_kwargs['reduce'] = True
        # self.final_attention = InvariantAttention(
        #     self.latent_dim,
        #     self._make_score_net(),
        #     self._make_value_net(),
        #     **final_kwargs)

    def _make_score_net(self):
        layers = []
        dim = self.latent_dim
        mlp_dim = int(self.dilation*self.latent_dim)

        for _ in range(self.mlp_layers):
            layers.append(torch.nn.Linear(dim, mlp_dim))
            layers.extend(self._get_normalization_layers('score', mlp_dim))
            if self.dropout:
                layers.append(torch.nn.Dropout(self.dropout))
            layers.append(self._activation_layer())
            dim = mlp_dim

        layers.append(torch.nn.Linear(dim, 1))
        return torch.nn.Sequential(*layers)

    def _make_value_net(self, output_dim=None, operating_on_invariants=True):
        mlp_dim = int(self.dilation*self.latent_dim)

        layers = []

        dim = self.latent_dim
        if operating_on_invariants:
            dim = self.invariant_dims
            layers.extend(self._get_normalization_layers('invariant_value', dim))

        for _ in range(self.mlp_layers):
            layers.append(torch.nn.Linear(dim, mlp_dim))
            layers.extend(self._get_normalization_layers('value', mlp_dim))
            if self.dropout:
                layers.append(torch.nn.Dropout(self.dropout))
            layers.append(self._activation_layer())
            dim = mlp_dim

        if output_dim:
            layers.append(torch.nn.Linear(dim, output_dim))
        return torch.nn.Sequential(*layers)

    def _get_normalization_layers(self, location, dim, force=None):
        result = []

        if force:
            normalization_style = force
        else:
            normalization_style = getattr(self, '{}_normalization'.format(location))

        if normalization_style == 'none':
            return []

        kwargs = dict(dim=dim, **self.normalization_kwargs)

        result.append(self._NORMALIZATION_FUNCTIONS[normalization_style](**kwargs))
        return result

    def _activation_layer(self):
        return self._ACTIVATION_FUNCTIONS[self.activation]()

    def _make_block_invariant_net(self):
        return self._make_value_net(self.latent_dim, operating_on_invariants=False)

    def _make_scale_net(self):
        return self._make_value_net(1, False)

    def _get_layer_inputs(self, block_index, last_eqvar, last_invar):
        nonnorm = (last_eqvar, last_invar)
        if self.normalize_equivariant_values:
            eqvar_normalized = self.input_norm_layers[block_index](last_eqvar)
            norm = (eqvar_normalized, last_invar)
            return [nonnorm] + (self.rank - 1)*[norm]
        return self.rank*[nonnorm]

    def forward(self, data):
        data = AtomicDataDict.with_edge_vectors(data, False)
        index_ij = data[AtomicDataDict.EDGE_INDEX_KEY]
        (unique_index_is, neighbor_counts) = index_ij[0].unique(return_counts=True)
        N = unique_index_is.shape[0]
        max_neighbors = neighbor_counts.max()
        size = [N, max_neighbors]
        rectangular_i = torch.zeros(*size, dtype=index_ij.dtype, device=index_ij.device)
        rectangular_j = torch.zeros(*size, dtype=index_ij.dtype, device=index_ij.device)
        mask = torch.zeros(*size, dtype=torch.bool, device=index_ij.device)

        rectangular_i[:] = unique_index_is[:, None]

        target_j_indices = torch.arange(
            0, index_ij.shape[1], dtype=index_ij.dtype, device=index_ij.device)
        row_deficiencies = max_neighbors - neighbor_counts
        row_skips = torch.roll(torch.cumsum(row_deficiencies, 0), 1)
        row_skips[0] = 0
        particle_skips = torch.repeat_interleave(row_skips, neighbor_counts)
        target_j_indices += particle_skips
        rectangular_j.view(-1)[target_j_indices] = index_ij[1]
        mask.view(-1)[target_j_indices] = 1

        bond_invar = data[self.edge_invariant_field]
        bond_eqvar = data[self.edge_covariant_field]
        invar_shape = [N, max_neighbors, bond_invar.shape[-1]]
        eqvar_shape = [N, max_neighbors, 3]
        rect_invar = torch.zeros(*invar_shape, dtype=bond_invar.dtype, device=bond_invar.device)
        rect_eqvar = torch.zeros(*eqvar_shape, dtype=bond_eqvar.dtype, device=bond_eqvar.device)
        rect_invar.view(-1,  bond_invar.shape[-1])[target_j_indices] = \
            bond_invar*self.invariant_scale
        rect_eqvar.view(-1, 3)[target_j_indices] = bond_eqvar*self.equivariant_scale

        last_eqvar, last_invar = rect_eqvar, rect_invar

        last_eqvar = self.vector_upcast(last_eqvar)
        last_invar = self.embedding_projection(last_invar)

        for i in range(self.num_blocks):
            residual_eqvar, residual_invar = last_eqvar, last_invar

            inputs = self._get_layer_inputs(i, last_eqvar, last_invar)
            if self.tied_attention:
                last_eqvar, last_invar = self.tied_layers[i](inputs, mask=mask)
            else:
                if self.use_multivectors:
                    last_eqvar = self.eqvar_layers[i](inputs, mask=mask)
                    inputs = self._get_layer_inputs(i, last_eqvar, last_invar)

                last_invar = self.invar_layers[i](inputs, mask=mask)

            if self.block_nonlinearity:
                last_invar = self.block_nonlin_layers[i](last_invar)

            if self.residual:
                if self.use_multivectors or self.tied_attention:
                    last_eqvar = last_eqvar + residual_eqvar
                last_invar = last_invar + residual_invar

            if self.block_norm_layers and (i + 1 < self.num_blocks or self.normalize_last_block):
                last_invar = self.block_norm_layers[i](last_invar)

            if self.eqvar_norm_layers and (i + 1 < self.num_blocks or self.normalize_last_block):
                last_eqvar = self.eqvar_norm_layers[i](last_eqvar)

        # inputs = self._get_layer_inputs(self.num_blocks, last_eqvar, last_invar)
        # last_invar = self.final_attention(inputs, mask=mask)

        data[self.out_field] = last_invar[mask]
        return data
