Usage
=====

Streaming
---------

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, start streaming the camera using the ``leprun`` command.

.. code-block:: console

   (.venv) C:\Users\username> leprun


When you are finshed streaming, press the ``esc`` while the viewer window is active to terminate.

Recording
---------

To record a stream, use the ``-r`` flag.

.. code-block:: console

   (.venv) C:\Users\username> leprun -r


All generated data is saved to the directory *Lepton_Recordings* which itself is generated in the active directory. After the recording is terminated, data is rendered into a .mp4 video.

When you are finshed recording, press the ``esc`` while the viewing window is active to terminate. After termination, a background process will render the video. This may take several minutes depending on the length of the recording.

Help
----

You can use the ``-h`` flag to explore addtional flags and functionality.

.. code-block:: console

   (.venv) C:\Users\username> leprun -h

In Code
-------
To interact with the Lepton in code, use the :ref:`Stream Class <stream-class>`. Example usage is given in :ref:`Examples <examples>`.


