{{ fullname | replace("kpfpipe.", "") | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{% if modules %}
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
