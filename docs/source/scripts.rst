.. include:: globals.rst

|hiperobs| scripts
*******************

This page documents all the scripts provided by the |pkg| which are all part of 
:mod:`hcam_obsutils.scripts`. Calling commands is done in the same way as the HiPERCAM
pipeline. If you are new to observing with |hiper| or |ultra|, you should refer to the
:ref:`command-calling` section for more information on how to call these scripts.

From the terminal, the quickest way to get help on any of the scripts is by using e.g
``pydoc hcam_obsutils.scripts`` or ``pydoc hcam_obsutils.scripts.missbias``. This will 
give you a quick overview of the script and its usage. 
For more detailed information, you can refer to the documentation for each script, 
which is linked below.

.. contents:: Contents
    :local:
    
Scripts
========
The table below provides a high level summary of installed scripts. Clicking 
on the script name will take you to the detailed documentation for that script.

.. table::
   :widths: 10 30 

   +--------------------+--------------------+
   | Command            | Purpose            |
   |                    |                    |
   +====================+====================+
   | |missbias|         | |ol-missbias|      |
   +--------------------+--------------------+
   | |unique|           | |ol-unique|        |
   +--------------------+--------------------+
   | |ucam_qc|          | |ol-uqc|           |
   +--------------------+--------------------+
   | |ucam_gain_simple| | |ol-ugain_simple|  |
   +--------------------+--------------------+
   | |ucam_gain|        | |ol-ugain|         |
   +--------------------+--------------------+
   | |uspec_qc|         | |ol-uspec_qc|      |
   +--------------------+--------------------+
   | |ucam_zp|          | |ol-ucam_zp|       |
   +--------------------+--------------------+
   | |hcam_zp|          | |ol-hcam_zp|       |
   +--------------------+--------------------+
   | |uspec_zp|         | |ol-uspec_zp|      |
   +--------------------+--------------------+

.. _command-calling:

Parameter specification
=======================
The script provided by the |pkg| are distinct from standard unix commands in having a 'memory', which is implemented through storage of inputs in disk files for each command, and also in prompting you if you don't specify a parameter on the command line.

The parameter memory along with the use of backslashes '\\\\' to accept default values can save a huge amount of typing making for efficient operation once you get up to speed.

See the detailed instructions in the `HiPERCAM pipeline manual <https://hipercam.github.io/hipercam/commands.html#parameter-specification>`_ for more details.

Script documentation
====================
This section contains documentation auto-generated from the code of each script. Each command appears as a function (an implementation detail), followed by a highlighted line showing the parameters one can use on the command-line. Inputs in square brackets such as ``[source]`` are hidden by default; those in round brackets e.g. ``(plot)`` may or may not be prompted depending upon earlier inputs. It is always safest when first running a command simply to type its name and hit enter and let the command itself prompt you for input. Many commands have hidden parameters that can only be revealed by typing e.g. ``rtplot prompt``. These are usually parameters that rarely need changing, but you are sure sometimes to need to alter them.  See the :ref:`command-calling` section for details on how to specify command parameters.

In the one-line descriptions below, ``run`` refers to a complete run, containing multiple images, stored in a .fits file. ``frame`` refers to a single image from a run as might be extracted using the pipeline command `grab <https://hipercam.github.io/hipercam/commands.html#hipercam.scripts.grab>`_. These have file extension '.hcm' to distinguish them, although they are also FITS-format files.

Detailed script documentation
#############################

.. autoapifunction:: hcam_obsutils.scripts.missbias.missbias
.. autoapifunction:: hcam_obsutils.scripts.unique.unique
.. autoapifunction:: hcam_obsutils.scripts.ucam.qc.qc.ucam_qc
.. autoapifunction:: hcam_obsutils.scripts.ucam.qc.gain_simple.ucam_gain_simple
.. autoapifunction:: hcam_obsutils.scripts.ucam.qc.gain.ucam_gain
.. autoapifunction:: hcam_obsutils.scripts.uspec.qc.uspec_qc
.. autoapifunction:: hcam_obsutils.scripts.ucam.ucam_zeropoints.ucam_zeropoints
.. autoapifunction:: hcam_obsutils.scripts.hcam.hcam_zeropoints.hcam_zeropoints
.. autoapifunction:: hcam_obsutils.scripts.uspec.uspec_zeropoints.uspec_zeropoints