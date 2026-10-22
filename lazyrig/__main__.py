if __name__ == "__main__":
    from . import __version__
    print(f'LazyRig {__version__}: starting; loading application modules...', flush=True)
    from .server import main
    main()
