#!/usr/bin/env python3
"""
Script to automatically disable OpenAI instrumentation when LangChain instrumentation
is present to avoid duplicate telemetry.

This script detects when both langchain and openai instrumentations would be loaded
and automatically sets OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=openai to prevent conflicts.
"""

import os
import sys
from importlib.metadata import entry_points
from typing import Set


def _is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def _has_instrumentation_entry_point(entry_point_name: str) -> bool:
    """Check if an instrumentation entry point exists."""
    try:
        instrumentors = entry_points(group="opentelemetry_instrumentor")
        return any(ep.name == entry_point_name for ep in instrumentors)
    except Exception:
        return False


def _get_disabled_instrumentations() -> Set[str]:
    """Get the current set of disabled instrumentations."""
    disabled = os.environ.get("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "")
    if not disabled:
        return set()

    # Handle comma-separated list and strip whitespace
    return {item.strip() for item in disabled.split(",") if item.strip()}


def _set_disabled_instrumentations(disabled_set: Set[str]) -> None:
    """Set the OTEL_PYTHON_DISABLED_INSTRUMENTATIONS environment variable."""
    if disabled_set:
        os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = ",".join(sorted(disabled_set))
    else:
        # Remove the variable if set is empty
        os.environ.pop("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", None)


def _langchain_supports_callbacks() -> bool:
    """
    Check if LangChain supports LLM callbacks by verifying the callback infrastructure.

    Returns:
        True if LangChain has callback support, False otherwise.
    """
    try:
        # Check if langchain_core.callbacks module exists
        from langchain_core.callbacks import BaseCallbackHandler  # type: ignore

        # Verify BaseCallbackHandler has the required methods for LLM instrumentation
        if not hasattr(BaseCallbackHandler, 'on_chat_model_start'):
            return False
        if not hasattr(BaseCallbackHandler, 'on_llm_end'):
            return False

        # Check if BaseCallbackManager exists (used by langchain instrumentation)
        try:
            from langchain_core.callbacks import BaseCallbackManager  # type: ignore
        except ImportError:
            # Try alternative import path
            try:
                from langchain_core.callbacks.base import BaseCallbackManager  # type: ignore
            except ImportError:
                return False

        return True
    except ImportError:
        # langchain_core.callbacks doesn't exist or can't be imported
        return False
    except Exception:
        # Any other error means we can't verify callback support
        return False

def resolve_langchain_openai_conflict() -> bool:
    """
    Detect and resolve the conflict between langchain and openai instrumentations.

    Returns:
        True if conflict was detected and resolved, False otherwise.
    """
    # Check if langchain instrumentation is available
    has_langchain_instrumentation = _has_instrumentation_entry_point("langchain")

    # Check if openai package is installed
    has_openai_package = _is_package_installed("openai")

    # Check if openai instrumentation is available
    has_openai_instrumentation = _has_instrumentation_entry_point("openai")

    # Conflict exists if:
    # 1. LangChain instrumentation is available
    # 2. OpenAI package is installed
    # 3. OpenAI instrumentation is available
    conflict_detected = (
            has_langchain_instrumentation
            and has_openai_package
            and has_openai_instrumentation
    )

    if not conflict_detected:
        return False

    # Get current disabled instrumentations
    disabled = _get_disabled_instrumentations()

    # Check if openai is already disabled
    if "openai" in disabled:
        return False  # Already disabled, no action needed

    # Add openai to disabled list
    disabled.add("openai")
    _set_disabled_instrumentations(disabled)

    print(
        "INFO: Detected conflict between langchain and openai instrumentations. "
        "Automatically disabling openai instrumentation to prevent duplicate telemetry. "
        f"OTEL_PYTHON_DISABLED_INSTRUMENTATIONS={os.environ.get('OTEL_PYTHON_DISABLED_INSTRUMENTATIONS')}",
        file=sys.stderr
    )

    return True