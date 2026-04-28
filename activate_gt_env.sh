#!/bin/sh

# Source this file from the project root:
#   . ./activate_gt_env.sh

SELF_PATH="${BASH_SOURCE:-$0}"

if [ ! -f "$SELF_PATH" ] && [ -f "./activate_gt_env.sh" ]; then
    SELF_PATH="./activate_gt_env.sh"
fi

case "$SELF_PATH" in
    /*) SCRIPT_PATH="$SELF_PATH" ;;
    *) SCRIPT_PATH="$PWD/$SELF_PATH" ;;
esac

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)

if [ ! -f "$ROOT_DIR/.venv/bin/activate" ]; then
    echo "Missing virtual environment at $ROOT_DIR/.venv"
    echo "Create it with: python3 -m venv --system-site-packages .venv"
    return 1 2>/dev/null || exit 1
fi

. "$ROOT_DIR/.venv/bin/activate"
PATH="$ROOT_DIR/.venv/bin:$PATH"
export PATH

export TAMP_PDDL_ROOT="$ROOT_DIR"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT/platforms"
export QT_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_LOGGING_RULES="${QT_LOGGING_RULES:-*.debug=false;qt.qpa.*=false}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/pddlstream${PYTHONPATH:+:$PYTHONPATH}"

# When running a GUI session from Wayland/XWayland, let Qt use the xcb backend
# but avoid forcing a particular GL integration plugin unless the user asks.
if [ -n "${DISPLAY:-}" ] && [ "${HEADLESS:-False}" != "True" ]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi

case ":${LD_LIBRARY_PATH:-}:" in
    *:"$COPPELIASIM_ROOT":*)
        ;;
    *)
        export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
esac

echo "Activated TAMP-PDDL ground-truth environment"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
echo "  COPPELIASIM_ROOT=$COPPELIASIM_ROOT"
echo "  DISPLAY=${DISPLAY:-<empty>}"
echo "  XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<empty>}"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
GT_PYTHON="$VENV_PYTHON"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "Warning: $VENV_PYTHON is not executable."
    echo "Rebuild the venv with: python3 -m venv --system-site-packages .venv"
else
    echo "  PYTHON=$VENV_PYTHON"
    if ! "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ['numpy', 'pyrep']
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
    then
        echo "Warning: the venv Python is missing required packages."
        echo "Install them with:"
        echo "  $VENV_PYTHON -m pip install numpy scipy pillow"
        if [ -x /usr/bin/python3 ] && /usr/bin/python3 - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('pyrep') else 1)
PY
        then
            GT_PYTHON="/usr/bin/python3"
            echo "Note: /usr/bin/python3 can already see 'pyrep'."
            echo "Use it for GT runs from this shell:"
            echo "  \$GT_PYTHON variation_1_easy/ground_truth_orchestrator_variation1_easy.py"
            echo "If you want the venv itself to work the same way, rebuild it with:"
            echo "  rm -rf .venv && /usr/bin/python3 -m venv --system-site-packages .venv"
        fi
    fi
fi
export GT_PYTHON
echo "  GT_PYTHON=$GT_PYTHON"

if [ -z "${DISPLAY:-}" ]; then
    echo "Warning: DISPLAY is empty. GUI CoppeliaSim runs will fail in this shell."
    echo "Use a normal X11 terminal session and source this file there."
fi
