"""Pipeline validation framework: instrument, measure, and report failures.

Runs the existing pipeline on real videos, saves every intermediate, computes
per-stage metrics, and produces one report on where the pipeline fails. Does
not modify the pipeline itself.
"""
