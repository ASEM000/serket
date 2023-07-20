``PyTreeClass`` exported API
=============================


.. currentmodule:: serket 

.. autoclass:: TreeClass 
.. autoclass:::members: at
.. autofunction:: is_tree_equal 
.. autofunction:: field
.. autofunction:: fields

``PyTreeClass`` exported pretty printing API
----------------------------------------------

.. currentmodule:: serket 

.. autofunction:: tree_diagram
.. autofunction:: tree_graph
.. autofunction:: tree_mermaid 
.. autofunction:: tree_repr 
.. autofunction:: tree_str
.. autofunction:: tree_summary
.. autofunction:: tree_repr_with_trace
.. currentmodule:: serket 
.. autofunction:: is_nondiff
.. autofunction:: freeze
.. autofunction:: unfreeze
.. autofunction:: is_frozen
.. autofunction:: tree_mask
.. autofunction:: tree_unmask


``PyTreeClass`` exported advanced API
---------------------------------------
.. currentmodule:: serket

.. autofunction:: bcmap
.. autoclass:: Partial
.. autoclass:: AtIndexer
    :members:
        get,
        set,
        apply,
        scan,
        reduce
.. autoclass:: BaseKey
    :members:
        __eq__
.. autofunction:: tree_map_with_trace
.. autofunction:: tree_leaves_with_trace
.. autofunction:: tree_flatten_with_trace

