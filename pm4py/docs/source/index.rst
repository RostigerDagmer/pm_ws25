Welcome to PM4Py's Documentation!
===================================

``PM4Py`` is a Python library implementing a variety of `process mining <https://en.wikipedia.org/wiki/Process_mining>`_ algorithms.

A simple example of ``PM4Py`` in action:

.. code-block:: python

   import pm4py

   if __name__ == "__main__":
       log = pm4py.read_xes('<path-to-xes-log-file.xes>')
       process_model = pm4py.discover_bpmn_inductive(log)
       pm4py.view_bpmn(process_model)
						
In this documentation, you can find all relevant information to set up ``PM4Py`` and start your process mining journey. 
Please consult the contents listed below to navigate the documentation.

Happy #ProcessMining!


Contents
--------

.. toctree::
   :maxdepth: 2

   install
   getting_started
   api
   release_notes
