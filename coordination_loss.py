import torch
from graph_pes.atomic_graph import neighbour_distances
from graph_pes.training.loss import Loss

class CoordinationWeightedMSE(Loss):
    def __init__(self, defect_weight=3.0, coordination_cutoff=2.2, weight=1000.0):
        super().__init__(weight=weight, is_per_atom=True)
        if defect_weight < 1 or coordination_cutoff <= 0:
            raise ValueError('Require defect_weight >= 1 and a positive cutoff')
        self.defect_weight = float(defect_weight)
        self.coordination_cutoff = float(coordination_cutoff)

    @property
    def required_properties(self):
        return ['tensor']

    @property
    def name(self):
        return 'coordination_mse'

    def forward(self, model, graph, predictions):
        pred, target = predictions['tensor'], graph.properties['tensor']
        if pred.shape != target.shape or target.shape != graph.Z.shape:
            raise ValueError('Expected one scalar shielding per atom')
        error2 = (pred - target).square()
        # Preserve ordinary validation loss for checkpoint selection.
        if not model.training:
            return error2.mean()
        with torch.no_grad():
            if float(graph.cutoff) < self.coordination_cutoff:
                raise ValueError('Model neighbour cutoff is too short')
            centre, neighbour = graph.neighbour_list
            z = graph.Z
            sio = ((z[centre] == 14) & (z[neighbour] == 8)) | ((z[centre] == 8) & (z[neighbour] == 14))
            close = neighbour_distances(graph) < self.coordination_cutoff
            counts = torch.bincount(centre[sio & close], minlength=len(z))
            defect = ((z == 14) & (counts != 4)) | ((z == 8) & (counts != 2))
            weights = torch.ones_like(error2)
            weights[defect] = self.defect_weight
        return (weights * error2).sum() / weights.sum()
