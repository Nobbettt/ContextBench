Installation
============

Requirements
------------

- Python 3.8 or higher
- Git (for cloning repositories)
- Required Python packages (see below)

Install from Source
-------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/EuniAI/ContextBench.git
   cd ContextBench

Install Dependencies
--------------------

**Recommended**: Install pinned runtime dependencies:

.. code-block:: bash

   pip install -r requirements.txt

**Critical**: Tree-sitter is required for symbol extraction. Ensure you have the correct versions:

.. code-block:: bash

   pip install "tree-sitter==0.20.4" tree-sitter-languages

Verify Installation
-------------------

Test that ContextBench is correctly installed:

.. code-block:: bash

   python -m contextbench.evaluate --help

You should see the command-line help message for the evaluation module.

Optional: Development Installation
-----------------------------------

If you plan to contribute or modify the code:

.. code-block:: bash

   pip install -e .

This installs ContextBench in editable mode, allowing you to make changes without reinstalling.

Troubleshooting
---------------

**Tree-sitter installation issues**

If you encounter errors with tree-sitter, make sure you have a C compiler available:

- **Linux**: ``sudo apt-get install build-essential``
- **macOS**: ``xcode-select --install``
- **Windows**: Install Visual Studio Build Tools

**Permission errors**

If you encounter permission errors, consider using a virtual environment:

.. code-block:: bash

   python -m venv contextbench_env
   source contextbench_env/bin/activate  # On Windows: contextbench_env\\Scripts\\activate
   pip install -r requirements.txt
