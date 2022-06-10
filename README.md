# pyqrack.github.io
Official documentation for PyQrack

# Contribution guidelines
If you want to contribute to the official documentation of PyQrack please
follow the following guidelines.

## Local testing
If you want to locally test the documentation:
```
git clone git@github.com:vm6502q/pyqrack.github.io.git
cd pyqrack.github.io/
make html
chrome build/html/index.html
```
Open `index.html` in your preferred browser.


## Autodocumentation
The documentation for PyQrack is generated through `sphinx`. Any changes that
are made in the docstrings of the original code are reflected here. In case
you have additions or changes that are to be made for documentation, please
file an issue on PyQracks's [repository](https://github.com/vm6502q/pyqrack/).

## Formatting Style
### Docstrings & Comments
Any form of documentation must follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
### Formatter
Just use black.
