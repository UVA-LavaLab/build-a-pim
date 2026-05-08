This directory contains classes and files which can be used to
implement cores in PIM architectures. For now, this supports bank
level SIMD cores only. The cores provided herein contain significant
amounts of duplicate code, which is not an accident. Each core is
intended as a fully-functional, self-contained example
implementation for how to extend lib.cores.components.base::BaseCore
yourself. If you want to modify the behavior of any of the provided
cores, feel free to create subclasses of them.
