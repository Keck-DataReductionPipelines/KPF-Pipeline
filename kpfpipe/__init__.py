from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    'chips' : ['GREEN', 'RED'],
    'fibers' : ['SKY','SCI1','SCI2','SCI3','CAL'],
    'norder' : {'GREEN':35, 'RED':32}
}
