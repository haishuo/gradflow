#!/usr/bin/env python3
"""Generate the frozen U4-B scalar-advection OpenSBLI application."""

import argparse
import ast
import collections
import collections.abc
import copy
import fractions
import importlib
import inspect
import math
import sys

# SymPy 1.1 is OpenSBLI's declared dependency. These names moved in newer
# Python runtimes, so expose their historical locations before importing it.
for _name in ("Mapping", "MutableMapping", "Sequence", "Callable", "Iterable", "Set"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

try:
    import sympy.printing.c as sympy_printing_c
except ImportError:
    sympy_printing_c = None
import sympy

# OpenSBLI's pinned revision imports a module path removed in newer SymPy.
if sympy_printing_c is not None:
    sys.modules.setdefault("sympy.printing.ccode", sympy_printing_c)
if sympy.__version__ != "1.1":
    sympy_compatibility = importlib.import_module("sympy.core.compatibility")
    if not hasattr(sympy_compatibility, "exec_"):
        sympy_compatibility.exec_ = exec
else:
    # SymPy 1.1 emits ast.Name(id="False") for evaluate=False. Python 3.11+
    # correctly rejects that obsolete AST representation. Repair the AST only;
    # the symbolic expression and its evaluation policy remain unchanged.
    from sympy.parsing import sympy_parser

    _evaluate_false = sympy_parser.evaluateFalse

    class _BooleanNameRepair(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "False":
                return ast.copy_location(ast.Constant(value=False), node)
            if node.id == "True":
                return ast.copy_location(ast.Constant(value=True), node)
            return node

    def _evaluate_false_modern_python(source):
        return ast.fix_missing_locations(_BooleanNameRepair().visit(_evaluate_false(source)))

    sympy_parser.evaluateFalse = _evaluate_false_modern_python

from sympy import Matrix, S

from opensbli import *
from opensbli.utilities.helperfunctions import substitute_simulation_parameters


class ScalarAdvectionEigensystem:
    """One-wave characteristic system for the scalar flux f(phi)=phi."""

    def generate_eig_system(self, block):
        if block.ndim != 1:
            raise ValueError("U4-B scalar adapter is intentionally one-dimensional")
        self.ev = Matrix([[S.One]])
        self.LEV = Matrix([[S.One]])
        self.REV = Matrix([[S.One]])

    def apply_direction(self, direction):
        if direction != 0:
            raise ValueError("scalar adapter has only direction zero")
        return {0: self.ev}, {0: self.LEV}, {0: self.REV}, set(), S.One


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, required=True)
parser.add_argument(
    "--case",
    choices=("state_a", "state_b", "constant", "sine"),
    required=True,
)
args = parser.parse_args()

state_expressions = {
    "state_a": "0.4 + sin(2*pi*DataObject(x0)) + 0.1*cos(6*pi*DataObject(x0))",
    "state_b": "sin(6*pi*DataObject(x0)) + 0.15*cos(8*pi*DataObject(x0))",
    "constant": "0.37",
    "sine": "sin(2*pi*DataObject(x0))",
}

simulation_parameters = {
    "c0": "1.0",
    # A zero RK step leaves phi unchanged. Each generated spatial stage hence
    # evaluates the same semidiscrete residual, which is exported below.
    "dt": "0.0",
    "niter": "1",
    "block0np0": str(args.size),
    "Delta0block0": "1.0/block0np0",
}

ndim = 1
selector = "**{'scheme':'Weno'}"
equation = "Eq(Der(phi,t), -Conservative(c_j*phi,x_j,%s))" % selector
expanded = EinsteinEquation().expand(equation, ndim, "x", [], ["c_j"])

simulation = SimulationEquations()
simulation.add_equations(expanded)
constituent = ConstituentRelations()

block = SimulationBlock(ndim, block_number=0)
block.set_block_boundaries([[PeriodicBC(0, 0), PeriodicBC(0, 1)]])

local_dict = {
    "block": block,
    "GridVariable": GridVariable,
    "DataObject": DataObject,
}
x0 = parse_expr(
    "Eq(DataObject(x0), block.deltas[0]*block.grid_indexes[0])",
    local_dict=local_dict,
)
phi = parse_expr(
    "Eq(DataObject(phi), %s)" % state_expressions[args.case],
    local_dict=local_dict,
)
initial = GridBasedInitialisation()
initial.add_equations([x0, phi])

schemes = {}
lf_weno = LFWeno(
    order=5,
    physics=ScalarAdvectionEigensystem(),
    averaging=SimpleAverage([0, 1]),
    formulation="JS",
    flux_type="LLF",
    flux_split=True,
    epsilon=1.0e-29 / 12.0,
)
schemes[lf_weno.name] = lf_weno
rk = RungeKutta(3)
schemes[rk.name] = rk

output = iohdf5(
    arrays=[DataObject("phi"), DataObject("Residual0"), DataObject("x0")],
    iotype="Write",
    write_constants=True,
)

block.set_equations([copy.deepcopy(constituent), copy.deepcopy(simulation), initial])
block.setio([output])
block.set_discretisation_schemes(schemes)
block.discretise()

algorithm = TraditionalAlgorithmRK(block)
SimulationDataType.set_datatype(Double)
OPSC(algorithm, OPS_V2=True)
substitute_simulation_parameters(
    simulation_parameters.keys(), simulation_parameters.values()
)
