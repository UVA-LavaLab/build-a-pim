---
title: _Build-A-PIM_ --- Extending DRAMSim3 to Support Bank-level Access Latency Measurement for Rapid PIM Design Scaffolding
author: William Bradford
theme:
    name: catppuccin-mocha
---

# Review: What is _Build-A-PIM_?

_Build-A-PIM_ is a library designed to enhance PIM literacy and simulation
capability for developers used to classical von-Neumann architectures. The
library has two parts:
1. A PIM Memory Transaction Simulation Backend
2. A Data Mapping Backend
    - Allows an array or other structure stored in a high
      level language to be directly accessed for simulation purposes

```
                                 Build-A-PIM      
Outside Implementation     ┌─────────────────────┐
   ┌─────────────┐         │                     │
   │     Per-PIM-Core Memory Requests            │
                     │                           │
    ┌───────────┐    │     ┌──────────────────┐  │
    │           │  ──┴──►  │  DRAMSim3-based  │  │
    │ Your Core │          │    Simulation    │  │
    │           │  ◄─┬───  │     Backend      │  │
    └───────────┘    │     └──────────────────┘  │
                     │                           │
        ▲     Announcements of                   │
        │  Transaction Completion                │
        │                  ┌───────────────────┐ │
        │                  │                   │ │
        │                  │  Data Placement   │ │
        └────────────────► │  Mapping Backend  │ │
                           │                   │ │
                           └───────────────────┘ │
                             ────────────────────┘
```

<!-- end_slide -->

# Updates

## New Additions

- Added a data wrapper which exposes language-native data in GDL-width chunks and communicates when said data is available
- Added a new timing "BIL" or Bank Interface Latency, which is the time between sending a DRAM polling command along the bus and its subsequent return to the memory controller
    - Request from Akhil and Sabiha
- Added data getting and setting to the memory model
    - MemSystem.get(location) -> DataWrapper\[GDL size\]
    - MemSystem.set(location, data) -> DataWrapper\[GDL size\]
    - Also has MemSystem\[_\] support, but is very clunky for setting data
- Implemented our first simple core model
    - **Mostly** complete, has some rough edges
    - Currently functions like our PIMeval Bank-level model, with pipelining between reads/writes and arithmetic/logical operations, but no pipelining otherwise. (Simple 2-stage pipeline: E,M/W)

## Collaborations
- Collaborated with Boming (this project essentially finished his simulator from what I understand)
- Collaborating with Akhil and Sabiha to make this tool fit their use-case

<!-- end_slide -->

# Updates

## New Propositions

- New programming model: Functional Assembly Intermediate Representation (FAIR)
    - Expose *all* data by its address in memory or explicitly denote that this data is *not* stored in memory
    - This makes all loads/stores implicit
        - Can follow an implementation model similar to how DRAMsim3 implements precharges and activations
        - In reality, this would be implemented at the compiler level
        - This model lets us reason about accesses much more easily

- Hide the refresh expense of PIM mode switching by adding a third mode:
    - On switch from PIM -> Main memory mode, all PIM cores are marked as "dirty" in the MCs bitmap
    - PIM cores intercept read/write commands and trigger the appropriate refresh operation
    - This PIM core is then marked as "clean"
    - We can then parallelize per-bank to other, non-servicing banks based on queued instructions

<!-- end_slide -->

# Building a Core (WIP)



<!-- end_slide -->

# Going Forward

***Disclaimer:*** This list is **enormous**: you can probably expect 3-4 of these things to be done in the "near-future."

- The API could use some additional fine-tuning
    - It would be nice to have the call to access and the call to load the GDL
    as part of one API call
- Add some more on-the-rails simulation for PIM cores 
    - Maybe a general class that can easily interface with the memory system
- Finish adding memory timing parameter querying
- Achieve more accurate simulation results for Iterative Filter-Update and
LoBSTA
    - I would like to implement a simplified assembly-esque 
- Direct integration with CACTI for scratchpad implementation / timing
- Energy metrics are currently disabled for PIM memory transactions -> flesh
out this implementation

## Roadmap:

1. Implement data wrapper monad         \[\]
2. Implement state load / store         \[\]
2. Establish instruction standard       \[󰇙\]
3. Replicate LoBSTA Core                \[:\]
4. Add standardized core infrastructure \[ \]
5. Create UPMEM core/model              \[ \]
6. ...

<!-- end_slide -->

