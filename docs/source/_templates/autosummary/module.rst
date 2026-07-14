{{ fullname | replace("kpfpipe.", "") | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{# Suppress the recursive submodule listing for "curated" packages — those whose
   __init__ re-exports a public API via __all__ (masters, quicklook, qc_flags,
   checkpoints, diagnostics). For those, `.. automodule:: {{ fullname }}` above
   already documents the re-exported classes INLINE (e.g. `Bias`), so also
   recursing into the implementation files would double-list every class under
   its with-stem module name (`modules.masters.bias`). Structural packages with
   no __all__ (data_models, utils, …) fall through and still recurse, which is
   how their submodule pages get generated. Detected via the `__all__` dunder
   showing up in `members` when (and only when) the package defines it. #}
{% if modules and '__all__' not in members %}
{# `modules` holds short names relative to this package; rebuild the full dotted
   name before matching the exact-name skip list from conf.py (so, e.g.,
   kpfpipe.data_models.config is dropped while kpfpipe.utils.config is kept). #}
{% set ns = namespace(visible=[]) %}
{% for item in modules %}
{% if (fullname ~ "." ~ item) not in (skip_modules | default([])) %}
{% set ns.visible = ns.visible + [item] %}
{% endif %}
{% endfor %}
{% if ns.visible %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in ns.visible %}
   ~{{ item }}
{%- endfor %}
{% endif %}
{% endif %}
{% endblock %}
