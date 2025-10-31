# make_petri_atlas.py
import os
import json
import hashlib
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# graph / ML
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

try:
    import hdbscan

    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

# pm4py
from pm4py.visualization.petri_net import visualizer as pn_vis

# ----------------------------
# Configuration
# ----------------------------
OUT_DIR = Path("petri_atlas_out")
SVG_DIR = OUT_DIR / "svgs"
DATA_CSV = OUT_DIR / "models.csv"
ATLAS_HTML = OUT_DIR / "atlas.html"
N_JOBS = min(40, os.cpu_count() or 8)  # you said ~40 cores
SEED = 42

# Rendering params: keep SVG small but readable
PN_VIS_PARAMS = {
    "format": "svg",
    "rankdir": "LR",  # left->right for better aspect ratio
    "bgcolor": "transparent",
    "debug": False,
}
# UMAP params (sensible defaults for graphs)
UMAP_KW = dict(
    n_neighbors=20, min_dist=0.15, metric="euclidean", random_state=SEED
)

# Clustering choice: "hdbscan" (preferred) or "kmeans"
CLUSTERING = "hdbscan"

# Thumbnail display size (CSS only)
THUMB_W = 40  # px
THUMB_H = 30  # px


# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def hash_petri(net) -> str:
    """
    Deterministic content hash so repeated nets reuse the same SVG/embedding.
    Uses (places, transitions, arcs) – insensitive to internal object IDs.
    """
    places = sorted([p.name for p in net.places])
    trans = sorted(
        [t.label if t.label is not None else t.name for t in net.transitions]
    )
    arcs = sorted([(a.source.name, a.target.name) for a in net.arcs])
    blob = json.dumps(
        {"P": places, "T": trans, "A": arcs}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def pm_to_nx_bipartite(net):
    """
    Convert PM4Py Petri net to a directed bipartite NetworkX graph.
    Node attrs:
      - type: "place" or "transition"
      - label: str (transition label if available; place name otherwise)
    """
    G = nx.DiGraph()
    for p in net.places:
        G.add_node(p.name, type="place", label=p.name)
    for t in net.transitions:
        lab = t.label if t.label is not None else t.name
        G.add_node(t.name, type="transition", label=str(lab))
    for a in net.arcs:
        G.add_edge(a.source.name, a.target.name)
    return G


def spectral_signature(G, k=6):
    A = nx.to_numpy_array(G)
    if A.size == 0:
        return np.zeros(k)
    vals = np.linalg.eigvalsh(A)
    vals = np.sort(np.real(vals))[-k:]
    return np.pad(vals, (0, max(0, k - len(vals))))


def cluster_embeddings(X):
    if CLUSTERING == "kmeans":
        k = int(
            np.clip(X.shape[0] / 60, 10, 80)
        )  # heuristic ~50 clusters for 3000
        labels = KMeans(
            n_clusters=k, n_init="auto", random_state=SEED
        ).fit_predict(X)
    else:
        if not HAVE_HDBSCAN:
            raise RuntimeError(
                "CLUSTERING='hdbscan' but hdbscan not installed."
            )
        labels = hdbscan.HDBSCAN(
            min_cluster_size=15, min_samples=5, cluster_selection_epsilon=0.0
        ).fit_predict(X)
    return labels


def entropy(xs):
    if not xs:
        return 0.0
    vals, counts = np.unique(xs, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))


def petri_features(net):
    """Compute a fixed-length feature vector for a PM4Py Petri net."""
    G = nx.DiGraph()
    for p in net.places:
        G.add_node(p.name, type="place")
    for t in net.transitions:
        lab = t.label if t.label else t.name
        G.add_node(t.name, type="transition", label=lab)
    for a in net.arcs:
        G.add_edge(a.source.name, a.target.name)

    places = [n for n, d in G.nodes(data=True) if d["type"] == "place"]
    trans = [n for n, d in G.nodes(data=True) if d["type"] == "transition"]

    np_p = len(places)
    np_t = len(trans)
    n_arcs = G.number_of_edges()
    ratio_pt = np_p / max(np_t, 1)

    # Degrees
    def deg_stats(nodes, mode="in"):
        if not nodes:
            return (0, 0)
        degs = [
            G.in_degree(n) if mode == "in" else G.out_degree(n) for n in nodes
        ]
        return (np.mean(degs), np.std(degs))

    p_in_m, p_in_s = deg_stats(places, "in")
    p_out_m, p_out_s = deg_stats(places, "out")
    t_in_m, t_in_s = deg_stats(trans, "in")
    t_out_m, t_out_s = deg_stats(trans, "out")

    # Connectivity & density
    try:
        scc = nx.number_strongly_connected_components(G)
    except Exception:
        scc = 0
    density = nx.density(G)

    # Entropy of transition labels
    label_entropy = entropy([G.nodes[n].get("label", "") for n in trans])

    # Approx path metrics
    if G.number_of_nodes() < 200:
        try:
            lengths = dict(nx.all_pairs_shortest_path_length(G))
            avg_len = np.mean(
                [v for d in lengths.values() for v in d.values()]
            )
        except Exception:
            avg_len = 0.0
    else:
        avg_len = 0.0

    feats = np.array(
        [
            np_p,
            np_t,
            n_arcs,
            ratio_pt,
            p_in_m,
            p_in_s,
            p_out_m,
            p_out_s,
            t_in_m,
            t_in_s,
            t_out_m,
            t_out_s,
            scc,
            density,
            label_entropy,
            avg_len,
        ],
        dtype=float,
    )
    sig = spectral_signature(G, k=6)
    feats = np.concatenate([feats, sig])

    return feats


def embed_structural(pm_dataset, n_jobs=8):
    """Compute a 2D numpy array [n_models, n_features]."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    feats = Parallel(n_jobs=n_jobs)(
        delayed(petri_features)(m["pm"])
        for m in tqdm(pm_dataset, desc="Embedding structural features")
    )
    X = np.vstack(feats)
    X = np.nan_to_num(X)
    return X


# ----------------------------
# Main API call
# ----------------------------
@dataclass
class PMItem:
    idx: int
    net: object
    im: object
    fm: object


def render_svg(item: PMItem, out_dir: Path) -> str:
    """
    Render the Petri net to SVG once; reuse by content hash.
    Returns the relative path to the SVG file.
    """
    h = hash_petri(item.net)
    svg_path = out_dir / f"{item.idx:05d}_{h}.svg"
    if not svg_path.exists():
        gviz = pn_vis.apply(
            item.net, item.im, item.fm, parameters=PN_VIS_PARAMS
        )
        pn_vis.save(gviz, str(svg_path))
    return svg_path.name


def build_atlas(
    pm_dataset,
    out_dir: Path = OUT_DIR,
    n_jobs: int = N_JOBS,
    wl_h: int = 2,
    use_umap: bool = True,
):

    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "svgs")

    # Wrap dataset into PMItem list
    items = [
        PMItem(i, m["pm"], m["im"], m["fm"]) for i, m in enumerate(pm_dataset)
    ]

    # 1) Render SVGs (parallel, but don’t oversubscribe Graphviz too hard)
    # With 40 cores and small nets, n_jobs ~ 8-16 is usually optimal for Graphviz.
    svg_names = Parallel(n_jobs=min(n_jobs, 16), prefer="processes")(
        delayed(render_svg)(it, out_dir / "svgs")
        for it in tqdm(items, desc="Rendering SVGs")
    )

    # 3) WL embeddings
    X = embed_structural(pm_dataset, n_jobs=n_jobs)
    X = StandardScaler().fit_transform(X)

    # 4) 2D layout for atlas
    if use_umap:
        coords = umap.UMAP(**UMAP_KW).fit_transform(X)
    else:
        coords = PCA(n_components=2, random_state=SEED).fit_transform(X)

    # 5) Clustering
    labels = cluster_embeddings(X)

    # 6) Normalize coords to [0,1] square for CSS positioning
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    norm = (coords - mins) / span

    # 7) Persist metadata
    meta = []
    for i, (name, (x, y), lab) in enumerate(zip(svg_names, norm, labels)):
        meta.append(
            {
                "idx": i,
                "svg": f"svgs/{name}",
                "x": float(x),
                "y": float(y),
                "cluster": int(lab),
            }
        )

    with open(out_dir / "models.json", "w") as f:
        json.dump(meta, f)

    # 8) Write HTML atlas
    write_html_atlas(meta, out_dir)

    print(f"\nDone.\nSVGs: {out_dir/'svgs'}\nAtlas: {out_dir/'atlas.html'}")


def write_html_atlas(meta, out_dir: Path):
    # Distinct clusters and a color palette (CSS HSL by cluster id; -1 = noise)
    cluster_ids = sorted({m["cluster"] for m in meta})
    # Simple legend data
    legend = [
        {"cluster": int(c), "count": sum(m["cluster"] == c for m in meta)}
        for c in cluster_ids
    ]

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Petri Net Atlas</title>
<style>
  :root {{
    --thumb-w: {THUMB_W}px;
    --thumb-h: {THUMB_H}px;
  }}
  html, body {{
    margin: 0; height: 100%; overflow: hidden; background: #0b0c10; color: #eee; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }}
  #toolbar {{
    position: fixed; top: 8px; left: 8px; right: 8px; padding: 8px 12px; background: #11151aee; border-radius: 10px;
    display: flex; align-items: center; gap: 12px; z-index: 10; backdrop-filter: blur(6px);
  }}
  #legend {{ display: flex; gap: 8px; flex-wrap: wrap; font-size: 12px; }}
  .legend-item {{
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 6px; border-radius: 6px; background: #0f1216;
  }}
  .dot {{
    width: 12px; height: 12px; border-radius: 50%;
  }}
  #canvas {{
    position: absolute; inset: 0; cursor: grab;
  }}
  #inner {{
    position: absolute; left: 0; top: 0; transform-origin: 0 0;
  }}
  .thumb {{
    position: absolute; width: var(--thumb-w); height: var(--thumb-h); overflow: hidden; border-radius: 10px;
    background: #f9f9f9; border: 1px solid #27303a; box-shadow: 0 1px 2px #0008;
    display: flex; align-items: center; justify-content: center;
  }}
  .thumb:hover {{ outline: 2px solid #9ad; z-index: 2; }}
  .thumb img {{
    width: calc(var(--thumb-w) - 10px); height: calc(var(--thumb-h) - 10px); object-fit: contain;
  }}
  .badge {{
    position: absolute; top: 4px; left: 6px; padding: 2px 6px; border-radius: 6px; font-size: 11px; color: #111; background: #eef;
    mix-blend-mode: screen;
  }}
  #search {{
    flex: 1; padding: 6px 10px; background: #0f141a; color: #ddd; border: 1px solid #2b3641; border-radius: 8px;
  }}
  #info {{ font-size: 12px; opacity: 0.8; }}
  a, a:visited {{ color: #aee; }}
</style>
</head>
<body>
<div id="toolbar">
  <input id="search" placeholder="Filter by cluster id (e.g., 0,1,2) or -1 for noise; empty shows all"/>
  <div id="legend"></div>
  <div id="info"></div>
</div>
<div id="canvas">
  <div id="inner"></div>
</div>
<script>
const META = {json.dumps(meta)};
const W = 4000, H = 3000; // logical canvas size before zoom
const inner = document.getElementById('inner');
const info = document.getElementById('info');
const legendDiv = document.getElementById('legend');
const search = document.getElementById('search');

// Build legend
const clusters = [...new Set(META.map(m => m.cluster))].sort((a,b)=>a-b);
const counts = Object.fromEntries(clusters.map(c => [c, META.filter(m=>m.cluster===c).length]));
legendDiv.innerHTML = clusters.map(c => {{
  const hue = (c < 0) ? 0 : (c*57 % 360);
  return `<span class="legend-item"><span class="dot" style="background:hsl(${{hue}},70%,60%)"></span>Cluster ${{c}} (${{counts[c]}})</span>`;
}}).join('');

// Build thumbs
const nodes = META.map((m,i) => {{
  const d = document.createElement('div');
  d.className = 'thumb';
  const hue = (m.cluster < 0) ? 0 : (m.cluster*57 % 360);
  d.style.left = (m.x * (W-200)) + 'px';
  d.style.top  = (m.y * (H-150)) + 'px';
  d.style.borderColor = `hsl(${{hue}},55%,45%)`;
  const b = document.createElement('div');
  b.className = 'badge';
  b.textContent = m.cluster;
  b.style.background = `hsl(${{hue}},80%,80%)`;
  d.appendChild(b);
  const img = document.createElement('img');
  img.loading = 'lazy';
  img.src = m.svg;
  img.title = `idx=${{m.idx}} cluster=${{m.cluster}}`;
  d.appendChild(img);
  d.onclick = () => window.open(m.svg, '_blank');
  inner.appendChild(d);
  return {{el:d, m}};
}});

// Pan/zoom
let scale = 0.35, ox = 40, oy = 80, dragging = false, sx=0, sy=0, sox=0, soy=0;
function applyTransform() {{ inner.style.transform = `translate(${{ox}}px, ${{oy}}px) scale(${{scale}})`; }}
applyTransform();
const canvas = document.getElementById('canvas');
canvas.addEventListener('mousedown', e => {{ dragging = true; sx=e.clientX; sy=e.clientY; sox=ox; soy=oy; canvas.style.cursor='grabbing'; }});
window.addEventListener('mouseup', ()=>{{ dragging=false; canvas.style.cursor='grab'; }});
window.addEventListener('mousemove', e => {{
  if(!dragging) return;
  ox = sox + (e.clientX - sx);
  oy = soy + (e.clientY - sy);
  applyTransform();
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const prev = scale;
  const delta = Math.sign(e.deltaY) * -0.1;
  scale = Math.max(0.1, Math.min(2.5, scale + delta));
  // zoom towards pointer
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  ox = cx - (cx - ox) * (scale/prev);
  oy = cy - (cy - oy) * (scale/prev);
  applyTransform();
}}, {{passive:false}});

// Filter by clusters
function applyFilter() {{
  const raw = search.value.trim();
  const allow = new Set(raw ? raw.split(/[,\\s]+/).filter(Boolean).map(Number) : clusters);
  let visible = 0;
  nodes.forEach(n => {{
    const on = allow.has(n.m.cluster);
    n.el.style.display = on ? '' : 'none';
    if(on) visible++;
  }});
  info.textContent = visible + " / " + META.length + " models";
}}
search.addEventListener('input', applyFilter);
applyFilter();
</script>
</body>
</html>
"""
    (out_dir / "atlas.html").write_text(html, encoding="utf-8")
