from typing import Callable


def load_callable(spec: str) -> Callable:
    """Load a callable from a file path spec like '/path/to/file.py::function'.

    Raises a descriptive exception if loading fails or the object is not callable.
    """
    import importlib.util
    file_path, func_name = spec.split("::", 1)
    spec_obj = importlib.util.spec_from_file_location("_dynamic_module", file_path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    candidate = getattr(module, func_name)
    if not callable(candidate):
        raise TypeError(f"Loaded object '{func_name}' from {file_path} is not callable")
    return candidate


