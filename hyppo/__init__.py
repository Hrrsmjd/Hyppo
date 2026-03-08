from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hyppo")
except PackageNotFoundError:
    __version__ = "0.2.0"

if not __version__:
    __version__ = "0.2.0"
