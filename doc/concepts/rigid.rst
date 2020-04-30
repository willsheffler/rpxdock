.. _rigid_body_page:

Rigid Body Representations
=============================

**this is out of date, concepts are similar, but constructs in rpxdock are named and behave differently**


Basic element of structure is a "rigid" chunk of protein structure called a Rose. This "rigid" structure may actually be a family of related conformations, if a proper hierarchical decomposition of conformations can be made.

Kinematic model is Roses and SymElems placed by a "scene tree" type structure called a Bouquet. All DOFs, including transforms which place entities in the tree, helical symmetry DOFs, and Rose conformations, are managed by hierarchal nested grids called :ref:`Nests <sampling_nest>`. Here is a silly example:

Scheme Bouquet Illustration

.. image:: /img/Scheme_Bouquet_Unicycle_Illustration.png
   :width: 600px

Here is the proposed object structure, which provides [1] efficient representation of multiple identical bodies in different positions (symmetry, multiple threads, etc...), [2] Interactions indirected WRT rigid body position and conformation choice.


