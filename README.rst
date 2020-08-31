DeepShutdown
============

Scheduling Servers Shutdown in Grid Computing with Deep Reinforcement Learning


Install
-------


1. Make sure you have Batsim v3.1.0 installed and working. Otherwise, you must follow `Batsim installation <https://batsim.readthedocs.io/en/latest/installation.html>`_ instructions. Check the version of Batsim with:

.. code-block:: bash

    batsim --version

2. Install SimpleRL:

.. code-block:: bash

    git clone https://github.com/lccasagrande/SimpleRL.git
    cd SimpleRL
    pip install -e .[tf]

3. Install DeepShutdown:

.. code-block:: bash

    git clone https://github.com/lccasagrande/DeepShutdown.git
    cd DeepShutdown
    pip install -e .
