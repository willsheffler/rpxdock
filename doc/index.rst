.. _index_page:

Welcome to rpxdock documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   intro
   installation
   tutorials/index
   concepts/index
   applications
   todo
   apidoc/modules

.. contents:: :local:

.. _index_summary:


Summary
=============

Rpxdock (as well as the ambitious vaporware scheme, and the conceptually related rifdock) is a multi-scale model of protein structure suited to global search of conformation space. rpxdock utilizes a novel transform-based objective function which :ref:`retains some of the power of fullatom force-fields <rpx_accuracy>`, while avoiding a costly and difficult-to-optimize fullatom model. The rpxdock model is carefully crafted to allow both :ref:`pair <pair_decomposition_page>` and :ref:`hierarchical <hierarchical_sampling>` decomposition of all underlying DOFs, opening the door to new optimization techniques like :ref:`sampling_page` and :ref:`hierarchical_packing`. Rpxdock and related projects are currently in use in the baker lab, and :ref:`seem to perform well <applications_page>`.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

