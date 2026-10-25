"""
Perturbation families for the stress protocol.

Every family obeys the same contract:

    apply_perturbation(G, family, level, seed, ...) -> (H, provenance)

where H is a NEW graph with FRESH integer node ids and `provenance` records,
for every node of H, where it came from:

    origin[new_id] = old_id     the node survived (possibly moved)
    origin[new_id] = None       the node was INVENTED (densification): no correct
                                match exists, matching it is always an error
    origin_multi[new_id]        list of origins, for merges (many-to-one); any of
                                them counts as correct
    deleted_origins             nodes of G with no image in H: their counterpart
                                is unmatchable and belongs in the
                                "expected no match" bucket, not in FN

Fresh ids matter: they stop node identity leaking into any later join, and step
06 parses embedding tokens with int(word.split('_')[1]), so ids must be ints.

Geometric levels are expressed in units of `s`, the median nearest-neighbour node
spacing of the source graph, so they transfer to any other network. The caller
computes `s` and passes the resulting amplitude in metres.
"""

import networkx as nx
import numpy as np

FAMILIES = ("identity", "jitter", "warp", "densify", "composite")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _coords(G, nodes):
    return np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes], dtype=float)


def _role(G, n):
    return G.nodes[n].get("role", "core")


def _relabel(G, id_offset):
    """Copy G with fresh consecutive integer ids. Attributes are preserved."""
    nodes = list(G.nodes())
    new_of_old = {old: id_offset + i for i, old in enumerate(nodes)}
    H = nx.relabel_nodes(G, new_of_old, copy=True)
    return H, new_of_old


# ----------------------------------------------------------------------
# geometric families
# ----------------------------------------------------------------------

def _apply_jitter(H, rng, sigma_m):
    """B1: independent isotropic Gaussian positional error.

    No global transform can undo this, so a registration baseline (ICP/CPD)
    cannot help here - unlike a translation or a similarity transform.
    """
    nodes = list(H.nodes())
    d = rng.normal(scale=sigma_m, size=(len(nodes), 2))
    for i, n in enumerate(nodes):
        H.nodes[n]["x"] += float(d[i, 0])
        H.nodes[n]["y"] += float(d[i, 1])
    return {"sigma_m": sigma_m}


def _apply_warp(H, rng, amplitude_m, correlation_length_m):
    """B2: spatially-correlated smooth deformation field.

    Nearby nodes are displaced together, which is how real survey and
    digitisation error actually behaves. This is the family that hurts a
    nearest-neighbour baseline most: a whole neighbourhood slides while the
    network's internal structure stays intact, so structural evidence still
    identifies a node when proximity no longer does.

    The field is built from i.i.d. Gaussian vectors on a control grid of
    spacing `correlation_length_m`, smoothed with a Gaussian kernel of the same
    width, then rescaled so the RMS displacement equals `amplitude_m`.
    """
    nodes = list(H.nodes())
    xy = _coords(H, nodes)

    L = float(correlation_length_m)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    # pad by one correlation length so the field is not clamped at the border
    lo, hi = lo - L, hi + L
    nx_ctrl = max(2, int(np.ceil((hi[0] - lo[0]) / L)) + 1)
    ny_ctrl = max(2, int(np.ceil((hi[1] - lo[1]) / L)) + 1)
    gx = np.linspace(lo[0], hi[0], nx_ctrl)
    gy = np.linspace(lo[1], hi[1], ny_ctrl)
    CX, CY = np.meshgrid(gx, gy)
    ctrl = np.column_stack([CX.ravel(), CY.ravel()])

    g = rng.normal(size=(len(ctrl), 2))

    d2 = ((xy[:, None, :] - ctrl[None, :, :]) ** 2).sum(axis=-1)
    w = np.exp(-d2 / (2.0 * L * L))
    w /= w.sum(axis=1, keepdims=True)
    disp = w @ g

    rms = float(np.sqrt((disp ** 2).sum(axis=1).mean()))
    disp = disp * (amplitude_m / rms) if rms > 0 else disp * 0.0

    for i, n in enumerate(nodes):
        H.nodes[n]["x"] += float(disp[i, 0])
        H.nodes[n]["y"] += float(disp[i, 1])

    return {
        "amplitude_rms_m": amplitude_m,
        "correlation_length_m": L,
        "control_points": int(len(ctrl)),
    }


# ----------------------------------------------------------------------
# topological families
# ----------------------------------------------------------------------

def _apply_densify(H, rng, edge_fraction, next_id):
    """C1: insert extra vertices along pipes.

    Models the very common case where one source stores geometry vertices and
    the other stores only junctions. This is the hardest family for a purely
    spatial method: an inserted vertex sits BETWEEN two real nodes, i.e. closer
    to the query than the true match, so nearest-neighbour and
    Hungarian-on-distance are drawn onto a decoy that has no correct partner.

    Dataset5 holds 1.26x the nodes of Dataset1, so edge_fraction ~ 0.26 is the
    level measured on the real pair rather than an invented one.
    """
    edges = list(H.edges())
    k = int(round(edge_fraction * len(edges)))
    k = max(0, min(k, len(edges)))
    if k == 0:
        return [], {"edge_fraction": edge_fraction, "inserted": 0}

    chosen = rng.choice(len(edges), size=k, replace=False)

    invented = []
    nid = next_id
    for i in chosen:
        u, v = edges[int(i)]
        xu, yu = H.nodes[u]["x"], H.nodes[u]["y"]
        xv, yv = H.nodes[v]["x"], H.nodes[v]["y"]
        # an inserted vertex is core only if it lies between two core nodes
        role = "core" if (_role(H, u) == "core" and _role(H, v) == "core") else "halo"
        H.remove_edge(u, v)
        H.add_node(nid, x=(xu + xv) / 2.0, y=(yu + yv) / 2.0, role=role)
        H.add_edge(u, nid)
        H.add_edge(nid, v)
        invented.append(nid)
        nid += 1

    return invented, {"edge_fraction": edge_fraction, "inserted": len(invented)}


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------

def _resolve_level(value, variant_level):
    """A composite step may pin its own level or inherit the variant's."""
    if isinstance(value, str) and value.strip() == "@level":
        return float(variant_level)
    return float(value)


def apply_perturbation(G, family, level, seed, node_spacing_m,
                       id_offset=10_000_000, params=None):
    """Apply one family - or an ordered composite of families - to G.

    `level` is in units of node spacing for the geometric families
    (jitter, warp) and is a fraction of edges for `densify`.

    COMPOSITE. Neither warp nor densify breaks a nearest-neighbour baseline on
    its own: warp moves a node less than the distance to its nearest decoy, and
    densify inserts decoys without moving anything, so the true match stays at
    distance 0. Together they do: densification roughly halves the spacing
    between candidates, which makes a given displacement twice as damaging.
    Steps are applied in the given order, and densify should come FIRST so the
    inserted vertices are carried by the same displacement field as everything
    else - inserting them afterwards would place them at un-warped positions.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family '{family}' (known: {FAMILIES})")

    params = dict(params or {})
    rng = np.random.default_rng(seed)

    H, new_of_old = _relabel(G, id_offset)
    origin = {new: old for old, new in new_of_old.items()}
    before = {n: (H.nodes[n]["x"], H.nodes[n]["y"]) for n in H.nodes()}

    if family == "composite":
        steps = params.get("steps") or []
        if not steps:
            raise ValueError("composite family requires params['steps']")
    else:
        steps = [{"family": family, "level": level, "params": params}]

    invented = []
    next_id = id_offset + len(new_of_old)
    applied = []

    for step in steps:
        fam = step["family"]
        lv = _resolve_level(step.get("level", "@level"), level)
        sp = dict(step.get("params") or {})

        if fam == "identity":
            rec = {}
        elif fam == "jitter":
            rec = _apply_jitter(H, rng, sigma_m=lv * node_spacing_m)
        elif fam == "warp":
            rec = _apply_warp(
                H, rng,
                amplitude_m=lv * node_spacing_m,
                correlation_length_m=sp.get("correlation_length_m", 500.0),
            )
        elif fam == "densify":
            made, rec = _apply_densify(H, rng, edge_fraction=lv, next_id=next_id)
            invented.extend(made)
            next_id += len(made)
        else:
            raise ValueError(f"'{fam}' cannot be used as a composite step")

        applied.append({"family": fam, "level": lv, **rec})

    for n in invented:
        origin[n] = None

    displacement = {}
    for n, (x0, y0) in before.items():
        displacement[n] = float(np.hypot(H.nodes[n]["x"] - x0, H.nodes[n]["y"] - y0))

    provenance = {
        "family": family,
        "level": level,
        "seed": seed,
        "node_spacing_m": node_spacing_m,
        "params": applied,
        "id_offset": id_offset,
        "origin": origin,
        "origin_multi": {n: ([o] if o is not None else []) for n, o in origin.items()},
        "inverse": {o: n for n, o in origin.items() if o is not None},
        "invented": list(invented),
        "deleted_origins": [],          # no family here deletes yet (block deletion will)
        "roles": {n: _role(H, n) for n in H.nodes()},
        "displacement": displacement,
    }
    return H, provenance
