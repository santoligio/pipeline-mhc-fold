#!/usr/bin/env python3
"""Shared, dependency-free helpers for the definitive redundancy analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}
MISSING = {"", ".", "?"}
PEPTIDE_TYPE_RE = re.compile(r"(?:^|[- ])PEPTIDE(?:[- ]|$)", re.I)


@dataclass(frozen=True)
class Residue:
    chain_id: str
    resseq: str
    insertion_code: str
    resname: str
    record_type: str


@dataclass(frozen=True)
class ChemComp:
    comp_id: str
    comp_type: str
    parent_id: str

    @property
    def is_peptide(self) -> bool:
        return bool(PEPTIDE_TYPE_RE.search(self.comp_type))


def log(message: str) -> None:
    print(message, flush=True)


def normalize_pdb_id(value: object) -> str:
    return str(value).split("-")[0].strip().upper()


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_id(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def cif_tokens(path: Path) -> Iterator[str]:
    """Yield STAR/mmCIF tokens, including quoted and semicolon text values."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = iter(handle)
        for raw_line in lines:
            if raw_line.startswith(";"):
                chunks = [raw_line[1:].rstrip("\r\n")]
                for text_line in lines:
                    if text_line.startswith(";"):
                        break
                    chunks.append(text_line.rstrip("\r\n"))
                yield "\n".join(chunks)
                continue
            line = raw_line.rstrip("\r\n")
            index = 0
            while index < len(line):
                while index < len(line) and line[index].isspace():
                    index += 1
                if index >= len(line) or line[index] == "#":
                    break
                quote = line[index] if line[index] in {"'", '"'} else None
                if quote:
                    index += 1
                    start = index
                    while index < len(line):
                        if line[index] == quote and (
                            index + 1 == len(line) or line[index + 1].isspace()
                        ):
                            yield line[start:index]
                            index += 1
                            break
                        index += 1
                    else:
                        raise ValueError(f"unterminated quoted token in {path}")
                    continue
                start = index
                while index < len(line) and not line[index].isspace():
                    index += 1
                token = line[start:index]
                if token:
                    yield token


class TokenStream:
    def __init__(self, tokens: Iterable[str]):
        self.tokens = iter(tokens)
        self.buffer: List[str] = []

    def push(self, token: str) -> None:
        self.buffer.append(token)

    def next(self) -> str:
        if self.buffer:
            return self.buffer.pop()
        return next(self.tokens)


def read_cif_loop(path: Path, category_prefix: str) -> Tuple[List[str], List[dict]]:
    """Read the first loop in an mmCIF category without external libraries."""
    stream = TokenStream(cif_tokens(path))
    try:
        while True:
            token = stream.next()
            if token.lower() != "loop_":
                continue
            headers: List[str] = []
            token = stream.next()
            while token.startswith("_"):
                headers.append(token)
                token = stream.next()
            if not headers or not headers[0].startswith(category_prefix):
                stream.push(token)
                continue
            values: List[str] = []
            while True:
                lower = token.lower()
                if token.startswith("_") or lower in {"loop_", "stop_"} or lower.startswith("data_"):
                    break
                values.append(token)
                try:
                    token = stream.next()
                except StopIteration:
                    break
            if len(values) % len(headers):
                raise ValueError(
                    f"malformed {category_prefix} loop in {path}: "
                    f"{len(values)} values for {len(headers)} columns"
                )
            rows = [
                dict(zip(headers, values[start : start + len(headers)]))
                for start in range(0, len(values), len(headers))
            ]
            return headers, rows
    except StopIteration:
        return [], []


def read_chem_comps(path: Path) -> Dict[str, ChemComp]:
    headers, rows = read_cif_loop(path, "_chem_comp.")
    if not rows:
        return {}
    id_key = "_chem_comp.id"
    type_key = "_chem_comp.type"
    parent_key = "_chem_comp.mon_nstd_parent_comp_id"
    if id_key not in headers or type_key not in headers:
        raise ValueError(f"_chem_comp loop lacks id/type in {path}")
    result: Dict[str, ChemComp] = {}
    for row in rows:
        comp_id = row[id_key].strip().upper()
        parent = row.get(parent_key, "").strip().upper()
        if parent in MISSING:
            parent = ""
        result[comp_id] = ChemComp(comp_id, row[type_key].strip(), parent)
    return result


def extract_pdb_residues(path: Path) -> Dict[str, List[Residue]]:
    """Read ordered residues from the first PDB model, de-duplicating altloc atoms."""
    chains: Dict[str, List[Residue]] = defaultdict(list)
    seen = set()
    model_number = 1
    explicit_models = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = line[:6].strip().upper()
            if record == "MODEL":
                explicit_models = True
                try:
                    model_number = int(line[10:14].strip())
                except ValueError:
                    model_number += 1
                continue
            if record == "ENDMDL" and explicit_models and model_number == 1:
                break
            if explicit_models and model_number != 1:
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            if len(line) < 27:
                raise ValueError(f"short coordinate record at {path}:{line_number}")
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip()
            resseq = line[22:26].strip()
            insertion_code = line[26].strip()
            if not chain_id or not resname or not resseq:
                raise ValueError(f"unreadable residue fields at {path}:{line_number}")
            key = (chain_id, resseq, insertion_code, resname)
            if key in seen:
                continue
            seen.add(key)
            chains[chain_id].append(
                Residue(chain_id, resseq, insertion_code, resname, record)
            )
    return dict(chains)


def peptide_token(resname: str, chem_comp: Optional[ChemComp]) -> str:
    """Canonical one-residue token; CCD parent maps PTMs to their parent amino acid."""
    if resname in AA3_TO_1:
        return AA3_TO_1[resname]
    if chem_comp and chem_comp.parent_id:
        parents = [part.strip() for part in chem_comp.parent_id.split(",")]
        mapped = [AA3_TO_1.get(parent) for parent in parents]
        if len(mapped) == 1 and mapped[0]:
            return mapped[0]
    return f"[{resname}]"


def is_peptide_residue(resname: str, chem_comp: Optional[ChemComp]) -> bool:
    return resname in AA3_TO_1 or bool(chem_comp and chem_comp.is_peptide)


def tokens_text(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def contiguous_relation(left: Sequence[str], right: Sequence[str]) -> Optional[str]:
    """Return exact/containment relation, requiring gapless contiguous matching."""
    left = tuple(left)
    right = tuple(right)
    if not left or not right:
        return None
    if left == right:
        return "exact"
    if len(left) < len(right) and find_subsequence(right, left) >= 0:
        return "left_in_right"
    if len(right) < len(left) and find_subsequence(left, right) >= 0:
        return "right_in_left"
    return None


def find_subsequence(longer: Sequence[str], shorter: Sequence[str]) -> int:
    if not shorter or len(shorter) > len(longer):
        return -1
    width = len(shorter)
    needle = tuple(shorter)
    for start in range(len(longer) - width + 1):
        if tuple(longer[start : start + width]) == needle:
            return start
    return -1


def match_sequence_multisets(
    left: Sequence[Sequence[str]], right: Sequence[Sequence[str]]
) -> Optional[List[Tuple[int, int, str]]]:
    """Find a perfect one-to-one exact/containment matching between chain multisets."""
    if len(left) != len(right):
        return None
    if not left:
        return []
    edges: Dict[int, List[Tuple[int, str]]] = {}
    for li, left_sequence in enumerate(left):
        candidates = []
        for ri, right_sequence in enumerate(right):
            relation = contiguous_relation(left_sequence, right_sequence)
            if relation:
                candidates.append((ri, relation))
        if not candidates:
            return None
        edges[li] = sorted(candidates, key=lambda item: (item[1] != "exact", item[0]))

    # Most constrained chains first prevents ambiguous duplicates from stealing matches.
    order = sorted(range(len(left)), key=lambda li: (len(edges[li]), li))
    matched_right: Dict[int, Tuple[int, str]] = {}

    def augment(li: int, visited: set) -> bool:
        for ri, relation in edges[li]:
            if ri in visited:
                continue
            visited.add(ri)
            if ri not in matched_right or augment(matched_right[ri][0], visited):
                matched_right[ri] = (li, relation)
                return True
        return False

    for li in order:
        if not augment(li, set()):
            return None
    return sorted((li, ri, relation) for ri, (li, relation) in matched_right.items())


def fetch_rcsb_metadata(
    pdb_ids: Sequence[str], cache_path: Path, refresh: bool = False,
    chunk_size: int = 100, retries: int = 4,
) -> Dict[str, dict]:
    """Fetch resolution/method in GraphQL batches and persist a reproducible cache."""
    cache: Dict[str, dict] = {}
    if cache_path.is_file():
        for row in read_csv(cache_path):
            cache[normalize_pdb_id(row["pdb_id"])] = dict(row)
    wanted = sorted(set(map(normalize_pdb_id, pdb_ids)))
    missing = wanted if refresh else [pdb_id for pdb_id in wanted if pdb_id not in cache]
    endpoint = "https://data.rcsb.org/graphql"
    # MSYS Python on Windows may not point OpenSSL at its installed CA bundle.
    # Verification stays enabled; we only locate an existing trusted bundle.
    ca_candidates = [
        os.environ.get("SSL_CERT_FILE", ""),
        ssl.get_default_verify_paths().cafile or "",
        str(Path(sys.prefix).parent / "usr" / "ssl" / "cert.pem"),
        str(Path(sys.executable).parents[2] / "usr" / "ssl" / "cert.pem"),
    ]
    cafile = next((candidate for candidate in ca_candidates if candidate and Path(candidate).is_file()), None)
    ssl_context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    for offset in range(0, len(missing), chunk_size):
        batch = missing[offset : offset + chunk_size]
        query = (
            "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
            "rcsb_entry_info{resolution_combined experimental_method}}}"
        )
        payload = json.dumps({"query": query, "variables": {"ids": batch}}).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "nefertari-redundancy/1.0"},
            method="POST",
        )
        response_data = None
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=45, context=ssl_context) as response:
                    response_data = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        if response_data is None:
            raise RuntimeError(f"RCSB request failed for {batch[0]}..{batch[-1]}: {last_error}")
        if response_data.get("errors"):
            raise RuntimeError(f"RCSB GraphQL error: {response_data['errors']}")
        returned = {}
        for entry in response_data.get("data", {}).get("entries", []) or []:
            pdb_id = normalize_pdb_id(entry["rcsb_id"])
            info = entry.get("rcsb_entry_info") or {}
            values = info.get("resolution_combined") or []
            numeric = [float(value) for value in values if value is not None]
            returned[pdb_id] = {
                "pdb_id": pdb_id,
                "resolution_angstrom": min(numeric) if numeric else "",
                "resolution_values": ";".join(map(str, numeric)),
                "experimental_method": info.get("experimental_method") or "",
                "api_status": "ok",
            }
        for pdb_id in batch:
            cache[pdb_id] = returned.get(pdb_id, {
                "pdb_id": pdb_id, "resolution_angstrom": "", "resolution_values": "",
                "experimental_method": "", "api_status": "not_returned",
            })
        log(f"[RCSB] metadata {min(offset + len(batch), len(missing))}/{len(missing)}")
    write_csv(cache_path, [
        "pdb_id", "resolution_angstrom", "resolution_values",
        "experimental_method", "api_status",
    ], (cache[pdb_id] for pdb_id in sorted(cache)))
    return cache


def load_cached_metadata(pdb_ids: Sequence[str], cache_path: Path) -> Dict[str, dict]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"offline mode requires resolution cache: {cache_path}")
    cache = {normalize_pdb_id(row["pdb_id"]): row for row in read_csv(cache_path)}
    missing = sorted(set(map(normalize_pdb_id, pdb_ids)) - set(cache))
    if missing:
        raise ValueError(f"resolution cache lacks {len(missing)} PDB IDs, e.g. {missing[:5]}")
    return cache
