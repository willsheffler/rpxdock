*******
RPXDock
*******

Version 2 of the venerable tcdock and rpxdock programs in rosetta. This version will be faster, simpler, and mostly in python.

Most of this code assumes all cyclic oligomers have their axis on Z (0,0,1) and dihedral oligmers have their highest order symmetry axis along Z and a twofold axis along X (1,0,0).

A full description on setup and installation, as well as a full description of relevant flags are in the RPXDock preprint: https://www.biorxiv.org/content/10.1101/2022.10.25.513641v1

Provided RPXDock examples:
* `tools/dock.sh` - simple docking example
* `tools/dump_pdb_from_output.py` - Output pdb from 
* `tools/rpxdock_to_design.sh` - simple example for 
* `tools/rpxdock_to_design.xml`

