import os

print(os.path.dirname(__file__))
print(os.path.join(os.path.dirname(__file__), ".."))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
