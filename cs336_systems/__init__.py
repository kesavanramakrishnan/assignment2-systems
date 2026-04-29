import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
<<<<<<< Updated upstream
    pass
=======
    __version__ = "0.0.0+local"
>>>>>>> Stashed changes
