"""The named non-foot prediction contract on the existing public Warp seam."""
import warp as wp

from .speculative_contact import require_pair_margin_backend, validate_prediction
from .task_proxy_contact import ContactBuffers, FrozenContactAdapter, buffers


@wp.kernel
def rebind_prediction(contact: ContactBuffers, count: wp.array[int],
                      address: wp.array2d[int], ground: int, targets: wp.array[int],
                      vel: wp.array2d[float], aref: wp.array2d[float],
                      pos: wp.array2d[float], D: wp.array2d[float],
                      errors: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    pair = contact.geom[i]
    gid = int(-1)
    if pair[0] == ground:
        gid = pair[1]
    elif pair[1] == ground:
        gid = pair[0]
    selected = bool(False)
    for j in range(targets.shape[0]):
        if gid == targets[j]:
            selected = True
    if not selected:
        return
    world = contact.worldid[i]
    row = address[i, 0]
    if row < 0:
        return
    if contact.dim[i] != 3 or D[world, row] <= 0.0:
        wp.atomic_add(errors, 0, 1)
        return
    gap = contact.dist[i]
    pos[world, row] = gap
    aref[world, row] = -vel[world, row] / 0.02 - gap / (0.02 * 0.02)


class SpeculativeContactAdapter(FrozenContactAdapter):
    def __init__(self, model, data, cpu_model, contract, *, entity_prefix=""):
        ground, targets = validate_prediction(cpu_model, contract, entity_prefix=entity_prefix)
        require_pair_margin_backend(contract, model)
        super().__init__(model, data, cpu_model, contract, entity_prefix=entity_prefix)
        if self.ground != ground:
            raise ValueError("Prediction ground identity changed")
        self.targets = wp.array(sorted(targets), dtype=int, device=self.device)

    def apply(self):
        # Parent owns sole quadrature, make_constraint and friction row units.
        super().apply()
        d = self.data
        wp.launch(rebind_prediction, d.naconmax,
            inputs=[buffers(d.contact), d.nacon, d.contact.efc_address, self.ground,
                    self.targets, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.D, self.errors],
            device=self.device)
