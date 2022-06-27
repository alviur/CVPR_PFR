import os

__all__ = list(
    map(lambda x: x[:-3],
        filter(lambda x: x not in ['__init__.py', 'learning_approach.py'] and x.endswith('.py'),
               os.listdir(os.path.dirname(__file__))
               )
        )
)
