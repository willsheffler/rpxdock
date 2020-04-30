Applications
=======================================

TODO: need current applications of actual rpxdock and maybe rifdock
The concepts described here also apply to the rifdock project. The MatDes docking methodology, sicdock, "Motif Designer" and "Motif Docking" in use in the Baker Lab amount to early prototypes of rpxdock.

Two Sided Design (easy) 
------------------------------------------


Cage Design
~~~~~~~~~~~~~~~~~~~~

Neil, Jacob, Yang Very early versions used in the Science and Nature papers, "motif docking" used in latest icosahedron set and Jacob seems to think it was an improvement.

Cyclic Oligomer Design
~~~~~~~~~~~~~~~~~~~~~~~~~~

Jorge: seems to be working fairly well, and with a very limited scaffold set.

Jeremy: 1 of 3 designs seems to work very well

Lattice Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ariel, soon to go to science or nature

Helical bundle design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

used for scoring

Possu, Gustav

Repeat Protein Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

used for scoring

TJ et al?

Heterodimer Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Daniel says it's fantastic and many of his hetero-dimer designs may work. we'll see if they're specific.


One-sided Binder design (hard)
------------------------------------------------

RIFdock
~~~~~~~~~~~~~~~~~~~

works well, see Jiayi's HBI binder paper in nature, all the IPD work on protien and small molecule binders in the past 2-3 years is rifdock??

MotifDock
~~~~~~~~~~~~~~~~~~~~~

*needs replacing with a rpxdock protocol!!*

Aaron says it works well and is using it. Luke, Kenwuan and David La have used it and said it was helpful, but I dunno if they're still using it.

Structure Prediction (0-sided, very hard)
---------------------------------------------

no data, need to do benchmarks!

Status 
---------

**This is all like 4 years out of date!!!! rpxdock and rifdock are now well estabilshed projects**

Prototype (formerly Sicdock) code and applications and "motif scoring" applications that people seem to think are good.

scheme is still mostly vapor, but we have:

 - [[Projects:rpxdock:Hierarchical Grid | Hierarchical Nested Grids]] interface is done-ish

  - I have updated the interface so that grids are now stateful. this is too bad, but I think it's required for different grids to have polymorphic effects. for example, different Grids might:

   - modify some transform in the fold tree
   - switch out one rose for another, e.g. in a hierarchical fragment grid
   - cache or compute a parametric helix
   - modify a Symmetry Element (helical symmetry elements are the only ones with DOFs ATM)

  - CompositeGrid working with BigNum arbitrary precision indexes

   - but this is not ideal for pair-decomposing the DOFs, must also have vector representation (see TODO)

  - some grids implemented

   - rectangular cartesian 1D, 2D 3D
   - Rotation
   - 2D Orientation (Unit Sphere)
   - 3D Orientation (Quad sphere hopf, probably need something better)
   - Rotation and translation along axis
   - Screw axis grids
   - Dependent grids for coupled degrees of freedom (symmetry)

 - Basic Gridded Kinematics

  - general object hierarchy

   - Bouquet

    -  Vase

     - Stem

      - SymElem
      - Rose (should be rosecontainer or rosegrid?)
      - Grid

    - SymmetricTopology

     - SymUpdateTrie

 - Nice [[Projects:rpxdock:Dynamic Symmetry Model|dynamic symmetry model]] is implemented in a modular way, and seems to be quite performant.
 