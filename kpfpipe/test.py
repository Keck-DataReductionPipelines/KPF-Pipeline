import os
import importlib
from modules.TemplateFit.KPFModule_dummy import KPFModule_dummy
def check_name(mod: str) -> bool:
    success = mod.endswith('.py')
    success &= mod.startswith('KPFModule_')
    return success

if __name__ == "__main__":
    for mod_folders in os.listdir('modules'):
        # loop through all files in this folder. 
        # If it's a .py that satisfies the naming convention, try to import it
        for modpy in os.listdir('modules/' + mod_folders):
            if check_name(modpy):
                mod = modpy.split('.')[0]
                import_path = '{}.{}.{}'.format('modules', mod_folders, mod)
                if mod == 'KPFModule_dummy':
                    res = importlib.import_module(import_path)
                    a = getattr(res, mod)
                    print(a)