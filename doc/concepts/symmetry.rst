.. _symmetry_page:

Dynamic Symmetry
=================================

Symmetry is represented by explicit symmetry elements which are placed in space in the same way that physical entities are. Symmetrically related coordinate frames are dynamically generated based on the placement of symmetry elements by (1) using a depth first search up to a specified depth to generate a symmetric topology and (2) each time the state of the system is changed, positions of the symmetric frames are updated based on the new symmetry element positions. Using a Trie based data structure to represent the symmetric topology makes this dynamic approach efficient enough that it is not a bottleneck, even for very complex crystal symmetries that require a deep search.


Below is a walleye stereo image of an I23 symmetry, colored by depth in the symmetry update trie. The trie-based update mechanism seems to be efficient enough to allow dynamically updating such deep symmetries as below without noticeable overhead.

I23 symmetry colored by SymUpdateTrie depth

.. image:: /img/Scheme_symmetry_I23_trie.png
   :width: 800px



