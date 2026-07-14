{% set _title = (title_overrides | default({})).get(fullname, fullname | replace("kpfpipe.", "")) %}
{{ _title | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{# Emit the submodule listing AFTER the automodule directive so a kept subpackage
   (e.g. Masters) sorts to the bottom of the page, below this package's inline
   class docs.

   For a *curated* package (its __init__ re-exports classes via __all__), those
   classes are documented inline by `.. automodule::` above, so listing the leaf
   submodules here too would double-list every class (once as `Bias`, once under
   its with-stem page `modules.masters.bias`) — drop them. True subpackages must
   survive that collapse and keep their own recursed page (e.g. kpfpipe.modules
   is curated, yet kpfpipe.modules.masters keeps its page); the `subpackages` set
   from conf.py identifies them. A *structural* package (no __all__) recurses into
   all its submodules as before. The conf.py `skip_modules` list drops internal
   modules in either case. Submodules are named with their short (leaf) names,
   which autosummary resolves against the current module set by the automodule
   directive above — hence this block must stay after it. #}
{% if modules %}
{% set curated = '__all__' in members %}
{% set ns = namespace(visible=[]) %}
{% for item in modules %}
{% set full = fullname ~ "." ~ item %}
{% if full not in (skip_modules | default([])) and ((not curated) or (full in (subpackages | default([])))) %}
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
