# Simumatik Student Sandbox
This repository is for the students doing their project

The sanbox python scripts (camera_sanbox.py and vr_sandbox.py) can be used with the local installation of the OEP server.
In order to use them, the files need to be copied to the local cache of the server, located at the user folder and named 'Simumatik', i.e. 'C:/Users/myself/Simumatik'.
The server, when is launched, will check if the files exist and will load them into memory. The scripts will be called at runtime when required.
Notice that the scripts are loaded just at startup, so changes won't make any effect until the server is restarted.
