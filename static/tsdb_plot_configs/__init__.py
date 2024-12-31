import os
import importlib

__all__ = []

# Get the absolute path to this directory
this_dir = os.path.dirname(__file__)

# A list with the paths to all yaml files is availble in 
#      static.tsdb_plot_configs.all_yaml
all_yaml = []

# Enumerate everything in the current directory
for name in os.listdir(this_dir):
    subdir_path = os.path.join(this_dir, name)
    
    # Check if it's a directory AND has an __init__.py (i.e., a subpackage)
    init_file = os.path.join(subdir_path, "__init__.py")
    if os.path.isdir(subdir_path) and os.path.isfile(init_file):
        # Dynamically import the subpackage
        module_name = f"static.tsdb_plot_configs.{name}"
        
        # Import the subpackage using importlib
        subpkg = importlib.import_module(module_name)
        
        # Make it available as an attribute of this package
        # e.g. tsdb_plot_configs.ccds => from ... import ccds
        globals()[name] = subpkg
        
        # Also, add it to __all__ so it’s included in "from ... import *"
        __all__.append(name)

        for file_in_subdir in os.listdir(subdir_path):
            if file_in_subdir.lower().endswith(".yaml"):
                yaml_path = os.path.join(subdir_path, file_in_subdir)
                all_yaml.append(yaml_path)
