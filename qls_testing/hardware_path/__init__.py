"""Hardware-oriented folded systems, block encodings, and readout plans."""

from .block_encoding import LCUBlockEncoding, build_lcu_block_encoding
from .folded_systems import (
    FoldedSystem,
    TerminalPaddedSystem,
    append_terminal_copies,
    build_folded_backward_euler,
    build_folded_bdf2,
    build_folded_crank_nicolson,
)
from .readout import ReadoutPlan, build_readout_plan, pennylane_overlap_quadratures
from .structure import CarlemanStructureReport, analyze_carleman_structure

__all__ = [
    "CarlemanStructureReport",
    "FoldedSystem",
    "TerminalPaddedSystem",
    "LCUBlockEncoding",
    "ReadoutPlan",
    "analyze_carleman_structure",
    "append_terminal_copies",
    "build_folded_backward_euler",
    "build_folded_bdf2",
    "build_folded_crank_nicolson",
    "build_lcu_block_encoding",
    "build_readout_plan",
    "pennylane_overlap_quadratures",
]
