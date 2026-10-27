from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

def _cpu_byte_rng_tensor(value: Any) -> torch.Tensor:
    """Convert a saved RNG state into the format PyTorch requires."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    return (
        value.detach()
        .to(
            device="cpu",
            dtype=torch.uint8,
        )
        .contiguous()
        .clone()
    )


def _json_safe(value: Any) -> Any:
    """Recursively convert scientific Python objects to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        # JSON NaN/Infinity are non-standard. Store missing non-finite metadata
        # as null rather than writing invalid cross-language JSON.
        return value if bool(np.isfinite(value)) else None

    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return _json_safe(detached.item())
        return _json_safe(detached.tolist())

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        converted = [_json_safe(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                allow_nan=False,
            ),
        )

    # Descriptive metadata objects such as torch.device and torch.dtype.
    return str(value)


"""Training utilities extracted from the publication notebook.

"""

def atomic_json_write(
    path: Path,
    payload: Any,
) -> None:
    """Write strict JSON atomically while preserving numeric tensor values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # Parse the complete temporary file before replacing the destination.
    json.loads(
        temporary.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON token: {token}")
        ),
    )
    os.replace(temporary, path)



def atomic_torch_save(
    payload: Any,
    path: Path,
) -> None:
    """Write one PyTorch file atomically and verify it can be loaded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    torch.save(payload, temporary)
    torch.load(
        temporary,
        map_location="cpu",
        weights_only=False,
    )
    os.replace(temporary, path)



def capture_rng_state(
    loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Capture portable CPU copies of every RNG state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": _cpu_byte_rng_tensor(
            torch.get_rng_state()
        ),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = [
            _cpu_byte_rng_tensor(cuda_state)
            for cuda_state
            in torch.cuda.get_rng_state_all()
        ]

    generator = (
        getattr(loader, "generator", None)
        if loader is not None
        else None
    )

    if generator is not None:
        state["loader_generator"] = (
            _cpu_byte_rng_tensor(
                generator.get_state()
            )
        )

    return state



def restore_rng_state(
    state: Optional[Dict[str, Any]],
    loader: Optional[DataLoader] = None,
) -> None:
    """Restore RNG states safely after CPU- or GPU-mapped loading."""
    if not state:
        return

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "numpy" in state:
        np.random.set_state(
            state["numpy"]
        )

    if "torch_cpu" in state:
        torch.set_rng_state(
            _cpu_byte_rng_tensor(
                state["torch_cpu"]
            )
        )

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        cuda_states = [
            _cpu_byte_rng_tensor(cuda_state)
            for cuda_state in state["torch_cuda"]
        ]

        device_count = torch.cuda.device_count()

        if len(cuda_states) == device_count:
            torch.cuda.set_rng_state_all(
                cuda_states
            )
        else:
            # This also supports restarting under a job
            # with a different number of allocated GPUs.
            for device_index, cuda_state in enumerate(
                cuda_states[:device_count]
            ):
                torch.cuda.set_rng_state(
                    cuda_state,
                    device=device_index,
                )

    generator = (
        getattr(loader, "generator", None)
        if loader is not None
        else None
    )

    if (
        generator is not None
        and "loader_generator" in state
    ):
        generator.set_state(
            _cpu_byte_rng_tensor(
                state["loader_generator"]
            )
        )


def save_history(
    history: Dict[str, Any],
    pt_path: Union[str, Path],
    json_path: Union[str, Path],
) -> None:
    atomic_torch_save(history, Path(pt_path))
    atomic_json_write(Path(json_path), history)
