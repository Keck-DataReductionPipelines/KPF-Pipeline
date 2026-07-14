{% set _title = (title_overrides | default({})).get(fullname, fullname.split(".") | last) %}
{{ _title | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{# The page/sidebar title defaults to the stemless leaf name (e.g.
   `barycentric_correction`, not `modules.barycentric_correction`); conf.py's
   `title_overrides` replaces it for named modules (both masters subpackages read
   as "Masters"). Each leaf module still resolves to its own recursed page.

   This submodule listing is emitted AFTER the automodule directive, and leaf
   modules are listed before subpackages, so a subpackage (e.g. Masters) sorts to
   the bottom. For a *curated* package (its __init__ re-exports classes via
   __all__), those classes are documented inline by `.. automodule::` above, so
   listing the leaf submodules here too would double-list every class (once as
   `Bias`, once under a with-stem page) — drop them, keeping only true
   subpackages, which the conf.py `subpackages` set identifies (so kpfpipe.modules
   recurses into per-module pages while curated kpfpipe.quality_control.quicklook
   collapses inline). A *structural* package (no __all__) recurses into all its
   submodules. The conf.py `skip_modules` list drops internal modules in either
   case. Submodules use their short (leaf) names, which autosummary resolves
   against the current module set by the automodule directive above — hence this
   block must stay after it. #}
{% if modules %}
{% set curated = '__all__' in members %}
{% set ns = namespace(leaves=[], subpkgs=[]) %}
{% for item in modules %}
{% set full = fullname ~ "." ~ item %}
{% set is_subpkg = full in (subpackages | default([])) %}
{% if full not in (skip_modules | default([])) and ((not curated) or is_subpkg) %}
{% if is_subpkg %}
{% set ns.subpkgs = ns.subpkgs + [item] %}
{% else %}
{% set ns.leaves = ns.leaves + [item] %}
{% endif %}
{% endif %}
{% endfor %}
{% set visible = ns.leaves + ns.subpkgs %}
{% if visible %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in visible %}
   ~{{ item }}
{%- endfor %}
{% endif %}
{% endif %}
{% endblock %}
