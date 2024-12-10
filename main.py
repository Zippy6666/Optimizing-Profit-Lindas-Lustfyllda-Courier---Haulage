from typing import Any


class DeliveryVan:
    def __init__(self):
        """Create a new delivery van."""


_delivery_vans = (DeliveryVan() for _ in range(10))
        

def get_delivery_van(package) -> Any:
    """
    Determines which van should haul the given package.

    Parameters
    ----------
    package : Any
        The package to check.

    Returns
    -------
    Any
        The van the package should be hauled with.
    """
    raise NotImplementedError()


def main():
    raise NotImplementedError()


if __name__ == "__main__":
    main()