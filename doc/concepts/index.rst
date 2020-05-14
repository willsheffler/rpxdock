Concepts
================================

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   rpx
   sampling
   rigid
   symmetry
   pair_decomp


Concepts Summary
-------------------------

Ridged body structure model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A molecular system is broken down into some number of rigid bodies. Depending on the system being modeled, these rigid units could be individual atoms, residues, fragments, whole chains or even assemblies of chains. These elements are placed in space using a transform based fold tree (like a scene graph).
:ref:`[More Info] <rigid_body_page>`

Dynamic Symmetry Model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symmetry is represented by explicit symmetry elements which are placed in space in the same way that physical entities are. Symmetrically related coordinate frames are dynamically generated based on the placement of symmetry elements.
:ref:`[More Info] <symmetry_page>`

RB Transform Based Scoring 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chemical entities are represented by full coordinate frames with a 3D position and orientation. The pair-score between two such entities is based on the full 6D transformation between the coordinate frames.
:ref:`[More Info] <rpx_page>`

Hierarchical Representations 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All sampling is done on an arbitrary precision, hierarchically nested grid. This allows iterative refinement in search using techniques like branch & bound. 
:ref:`[More Info] <sampling_page>`

Pair Decomposition 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All scores should be pair-decomposable. In combination with the gridded ridged body kinematic model, this allows any system with more than two rigid bodies to be optimized by precomputing pair-energy tables. By analogy to the packer, each rigid body is like a "residue" and each sampled position of a rigid body is like a "rotamer." 
:ref:`[More Info] <pair_decomposition_page>`


Ensemble Energy Evaluation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scores must be computed such that they evaluate an ensemble of conformations, not just a single one. Sampling is done in a hierarchy such that each sample point must cover a defined region of space. High up in the hierarchy, a sample point would be responsible for a large volume of space, say, 5Å in diameter with orientational deviation up to 15°. Lower down in the iterative sampling hierarchy, a sample point might represent a region 0.1Å in diameter and 1° orientational deviation.
:ref:`[More Info] <rpx_page_grids>`

Branch and Bound 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to construct score functions that rigorously bound the best possible score in a region of conformation space. If scoring can be set up in this manner, along with careful tracking of the volume sample points must represent, a branch and bound search is possible. Such a search would guarantee that no possible solution worse than a reported threshold was missed in the search.
:ref:`[More Info] <rpx_page_bounding>`
