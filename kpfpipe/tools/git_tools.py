import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_tag() -> str:
    return subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0']).decode('ascii').strip()

def get_git_branch() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
