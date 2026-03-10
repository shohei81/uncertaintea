# Documentation

This folder tracks the current design direction of UncertainTea.

## Contents

- [research.md](research.md): ecosystem research and the implications for a GPU-native Julia PPL
- [architecture.md](architecture.md): the current recommended architecture and layering strategy
- [dsl.md](dsl.md): a minimal DSL proposal that is intentionally closer to official Gen syntax

## Documentation Rules

- Update the relevant docs when a design decision changes
- Keep research notes dated and backed by primary-source links
- Update `dsl.md` and `architecture.md` together when the surface syntax changes
- Keep the boundary between the GPU static subset and CPU fallback explicit
