
from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.data import AtomicDataDict, AtomicDataset

from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)
from allegro._keys import EDGE_FEATURES, EDGE_ENERGY

from nequip.model import builder_utils
from ..nn import GAlA_Module
from ..nn.embedding import EdgeSymmetricEmbedding

def GAlA(config, initialize, dataset=None):

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    gala_kwargs = {}

    atomwise = False
    if 'atomwise' in config:
        atomwise = config['atomwise']
    elif 'gala_atomwise' in config:
        atomwise = config['gala_atomwise']

    if atomwise:
        gala_kwargs['out_field'] = AtomicDataDict.PER_ATOM_ENERGY_KEY

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot": OneHotAtomEncoding,
        'symmetric': EdgeSymmetricEmbedding,
        'gala': (
            GAlA_Module,
            gala_kwargs
        ),
    }

    if not atomwise:
        layers['edge_eng'] = (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY, mlp_output_dimension=1),
        )
        # Sum edgewise energies -> per-atom energies:
        layers["edge_eng_sum"] = EdgewiseEnergySum

    # Sum system energy:
    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )
    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
