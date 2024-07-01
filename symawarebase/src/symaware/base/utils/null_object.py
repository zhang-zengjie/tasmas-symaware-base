from typing import NoReturn, TypeVar, cast

Nullobjectself = TypeVar("Nullobjectself", bound="NullObject")


class NullObject:
    """
    Object implementing the null object pattern.
    It is used as a placeholder when an object is needed but no actual object is available.
    Using the object in any way will raise an exception.

    Example
    -------
    Using a :class:`.NullObject` results in an exception being raised.

    >>> from symaware.base.utils import NullObject
    >>> null_object = NullObject()
    >>> try:
    ...     null_object.attribute
    ... except NotImplementedError as e:
    ...     print(e)
    NullObject is a NullObject, it is used as a placeholder. Make sure to instantiate a valid instance of this class
    """

    _instance: "NullObject | None" = None

    @classmethod
    def instance(cls: "type[Nullobjectself]") -> "Nullobjectself":
        """
        Get the instance of the :class:`.NullObject` with a singleton pattern.

        Returns
        -------
            The singleton instance of the :class:`.NullObject`.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cast(Nullobjectself, cls._instance)

    def _null_error(self) -> NoReturn:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a NullObject, it is used as a placeholder. "
            "Make sure to instantiate a valid instance of this class"
        )

    def __getattribute__(self, name: str):
        if name == "_null_error" or name.startswith("__"):
            return super().__getattribute__(name)
        self._null_error()
