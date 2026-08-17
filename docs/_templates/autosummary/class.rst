{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

{% if attributes %}
Attributes
----------

.. autosummary::
{% for item in attributes %}
   ~{{ objname }}.{{ item }}
{%- endfor %}

{% for item in attributes %}
{% if is_property(module, objname, item) %}
.. autoproperty:: {{ objname }}.{{ item }}
{% else %}
.. autoattribute:: {{ objname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}

{% if methods %}
Methods
-------

.. autosummary::
{% for item in methods %}
{% if item != "__init__" %}
   ~{{ objname }}.{{ item }}
{% endif %}
{%- endfor %}

{% for item in methods %}
{% if item != "__init__" %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}
