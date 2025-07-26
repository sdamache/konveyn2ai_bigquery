"""
GuardFort Middleware Package

Provides authentication, logging, and request tracing middleware
for the KonveyN2AI three-component architecture.
"""

from .guard_fort import GuardFort, init_guard_fort

__version__ = "1.0.0"
__all__ = ["GuardFort", "init_guard_fort"]