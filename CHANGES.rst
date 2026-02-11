1.0.2 (2026-02-11)
==================

Bug Fixes
---------

- Fix two bugs in the ``LabelMapperConverter``:

     1. When roundtripping a ``LabelMapperArray``, with a non-default number
        of ``inputs``, deserialization fails because the init of the
        ``LabelMapperArray`` will set the number of inputs in the model to the
        default value of 2, which is then attempted to be overridden by later
        deseriaization methods.

     2. The converter fails when ``lazy_load`` is set to false and the
        ``mapper`` is an array.

  Both of these issues have been resolved. (`#701
  <https://github.com/spacetelescope/gwcs/issues/701>`_)


1.0.1 (2026-01-22)
==================

Bug Fixes
---------

- Fix gwcs evaluation when the input or output frames are ``None`` or
  ``EmptyFrame``. (`#684 <https://github.com/spacetelescope/gwcs/issues/684>`_)
