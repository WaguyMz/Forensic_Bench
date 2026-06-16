from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .context import get_default_graph_depth
from .db import _get_cursor

log = logging.getLogger(__name__)

_GRAPH: Any = None
_GRAPH_BUILT_AT: float = 0


def reset_graph() -> None:
    global _GRAPH, _GRAPH_BUILT_AT
    _GRAPH = None
    _GRAPH_BUILT_AT = 0


def _build_graph() -> Any:
    import networkx as nx

    G = nx.MultiDiGraph()
    with _get_cursor() as cur:
        cur.execute(
            "SELECT employee_id, display_name, payroll_account_number, cost_center FROM employees LIMIT 5000"
        )
        emp_rows = cur.fetchall()
        cur.execute(
            "SELECT vendor_id, name, primary_account_number, auxiliary_gl_account FROM vendors LIMIT 5000"
        )
        vnd_rows = cur.fetchall()
        cur.execute(
            "SELECT customer_id, name, primary_account_number, auxiliary_gl_account FROM customers LIMIT 5000"
        )
        cust_rows = cur.fetchall()

    for row in emp_rows:
        G.add_node(
            row[0],
            type="employee",
            name=row[1],
            bank_acct=row[2] or "",
            cost_center=row[3] or "",
        )
    for row in vnd_rows:
        G.add_node(
            row[0],
            type="vendor",
            name=row[1],
            bank_acct=row[2] or "",
            aux_gl=row[3] or "",
        )
    for row in cust_rows:
        G.add_node(
            row[0],
            type="customer",
            name=row[1],
            bank_acct=row[2] or "",
            aux_gl=row[3] or "",
        )

    log.info("Graph built: %d nodes", G.number_of_nodes())
    return G


def graph_query(
    start_id: str,
    entity_type: str,
    edge_types: Optional[List[str]] = None,
    depth: int = 2,
) -> str:
    global _GRAPH, _GRAPH_BUILT_AT
    _ = entity_type  # reserved for future filtering

    if _GRAPH is None or (time.time() - _GRAPH_BUILT_AT) > 300:
        try:
            _GRAPH = _build_graph()
            _GRAPH_BUILT_AT = time.time()
        except Exception as exc:
            return f"[GRAPH ERROR] Could not build graph: {exc}"

    G = _GRAPH
    depth = max(1, min(int(depth), get_default_graph_depth()))
    edge_filter: Optional[set[str]] = set(edge_types) if edge_types else None

    if start_id not in G:
        candidates = [n for n in G.nodes if str(n).startswith(start_id)]
        if not candidates:
            return f"[GRAPH] Entity '{start_id}' not found in graph."
        start_id = candidates[0]

    visited: Dict[str, int] = {start_id: 0}
    queue = [start_id]
    triples: List[str] = []

    while queue:
        node = queue.pop(0)
        node_depth = visited[node]
        if node_depth >= depth:
            continue
        for _, nbr, data in G.out_edges(node, data=True):
            etype = data.get("type", "related")
            if edge_filter is not None and etype not in edge_filter:
                continue
            src_meta = G.nodes[node]
            nbr_meta = G.nodes.get(nbr, {})
            triples.append(
                f"({node} [{src_meta.get('type','?')}:{str(src_meta.get('name',node))[:30]}]) "
                f"--[{etype}]--> "
                f"({nbr} [{nbr_meta.get('type','?')}:{str(nbr_meta.get('name',nbr))[:30]}])"
            )
            if nbr not in visited:
                visited[nbr] = node_depth + 1
                queue.append(nbr)

    if not triples:
        return f"[GRAPH] No edges found from '{start_id}' within depth {depth}."
    return (
        f"Subgraph from '{start_id}' (depth {depth}): {len(triples)} edge(s)\n\n"
        + "\n".join(triples)
    )
