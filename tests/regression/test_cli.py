import subprocess
from tests.regression.test_recipe import basics_recipe

def test_help():    
    cmd = 'kpf'
    child = subprocess.Popen([cmd], stdout=subprocess.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode

    assert rc == 2, "running command '{}' failed".format(cmd)

def test_launch():
    cmd = 'kpf -r examples/simple.recipe -c examples/default_simple.cfg'
    child = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    print(rc)
    assert rc == 1, "running command '{}' failed".format(cmd)
