Run Agents on ContextBench
==========================

ContextBench includes a unified runner for executing agents and collecting trajectories.

Runner entrypoint
-----------------

Use the module entrypoint:

.. code-block:: bash

   python -m contextbench.run --help

Common examples
---------------

.. code-block:: bash

   # Run agentless on Verified
   python -m contextbench.run --agent agentless --bench Verified

   # Run MiniSWE on Pro, first 5 instances
   python -m contextbench.run --agent miniswe --bench Pro --limit 5

Task lists
----------

By default the runner reads:

- ``data/selected_500_instances.csv``

See also
--------

- The Markdown guide: ``docs/run_agent_on_contextbench.md``
