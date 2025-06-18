from git import Repo, InvalidGitRepositoryError
import os


def get_git_revision_hash(repo_path: str = None) -> str:
    """
    Returns the current git commit hash for the repository at repo_path.
    If repo_path is None, uses the current working directory.
    """
    if repo_path is None:
        repo_path = os.getcwd()
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        return "Not a git repository"


def get_git_tag(repo_path: str = None) -> str:
    """
    Returns the most recent git tag for the repository at repo_path.
    If repo_path is None, uses the current working directory.
    """
    if repo_path is None:
        repo_path = os.getcwd()
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
        return tags[-1].name if tags else "No tags found"
    except InvalidGitRepositoryError:
        return "Not a git repository"


def get_git_branch(repo_path: str = None) -> str:
    """
    Returns the current branch name for the repository at repo_path.
    If repo_path is None, uses the current working directory.
    """
    if repo_path is None:
        repo_path = os.getcwd()
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.active_branch.name
    except (InvalidGitRepositoryError, TypeError, AttributeError):
        return "Not a git repository or detached HEAD"
