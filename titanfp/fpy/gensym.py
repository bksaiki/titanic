"""Unique name generation"""

class Gensym(object):
    """
    Unique name generator.
    
    Generates names of the form `<prefix><integer>`.
    The name is guaranteed to be unique among the names it
    has generated and any reserved names that are set during
    initialization of the generator.
    """
    names: set[str]
    counter: int

    def __init__(self, *names: str):
        self.names = set(names)
        self.counter = 1

    def __call__(self, prefix: str = 't'):
        """Generates a unique name with a given prefix."""
        name = f'{prefix}{self.counter}'
        while name in self.names:
            self.counter += 1
            name = f'{prefix}{self.counter}'

        self.names.add(name)
        return name
    
    def reserve(self, name):
        """Adds a reserved name."""
        self.names.add(name)
