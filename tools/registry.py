

import os
import importlib
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_TOOL_REGISTRY = {}
_initialized = False

def register_tool(name: str, tool_instance):
    if name in _TOOL_REGISTRY:
        logger.debug(f"Tool '{name}' already registered, skipping.")
        return
    _TOOL_REGISTRY[name] = tool_instance
    logger.info(f"Registered tool: {name}")

def get_all_registered_tools():
    return _TOOL_REGISTRY.copy()

def discover_and_register_tools(tools_dir: str = None, package: str = "tools"):
    global _initialized
    if _initialized:
        logger.debug("Tools already discovered and registered; skipping.")
        return

    if tools_dir is None:
        import tools
        tools_dir = os.path.dirname(tools.__file__)

    logger.info(f"Discovering tools in {tools_dir}...")

    for filename in os.listdir(tools_dir):
        if not filename.endswith(".py") or filename in ("__init__.py", "tool_registry.py"):
            continue

        module_name = filename[:-3]
        full_module_name = f"{package}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
            logger.info(f"Imported module {full_module_name}")

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and hasattr(attr, "name") and hasattr(attr, "args_schema"):
                    register_tool(attr.name, attr)

        except Exception as ex:
            logger.error(f"Error importing tools from {full_module_name}: {ex}")

    _initialized = True
    logger.info(f"Tool discovery completed. Total tools registered: {len(_TOOL_REGISTRY)}")