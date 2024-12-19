"""Unique name generation"""

class Gensym(object):
    """
    Unique name generator.
    
    Generates names of the form `<prefix><integer>`.
    The name is guaranteed to be unique among the names it
    has generated and any reserved names that are set during
    initialization of the generator.
    """

    _names: set[str]
    _counter: int

    def __init__(self, *names: str):
        self._names = set(names)
        self._counter = len(names)

    def reserve(self, *names: str):
        """Reserves a list of names."""
        self._names.update(names)

    def fresh(self, prefix: str = 't'):
        """Generates a unique name with a given prefix."""
        name = prefix
        while name in self._names:
            name = f'{prefix}{self._counter}'
            self._counter += 1

        self._names.add(name)
        return name

    def __contains__(self, name: str):
        return name in self._names

    def __len__(self):
        return len(self._names)
