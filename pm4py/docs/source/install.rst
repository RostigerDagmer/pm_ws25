```rst
Installation
===========

pip
---

To use ``PM4Py`` on any OS, install it using ``pip``:

.. code-block:: console

   (.venv) $ pip install pm4py

``PM4Py`` uses the ``Graphviz`` library for rendering visualizations.
Please install `Graphviz <https://graphviz.org/download/>`_.

After installation, Graphviz is located in the ``program files`` directory.
The ``bin\`` folder of the Graphviz directory needs to be added manually to the ``system path``.
In order to do so, please follow `this instruction <https://stackoverflow.com/questions/44272416/how-to-add-a-folder-to-path-environment-variable-in-windows-10-with-screensho>`_.

Docker
------
To install PM4Py via Docker, use:

.. code-block:: console

   $ docker pull pm4py/pm4py-core:latest

To run PM4Py via Docker, use:

.. code-block:: console

   $ docker run -it pm4py/pm4py-core:latest bash
```