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

<!-- column_layout: [3, 2] -->
<!-- column: 0 -->

Below is a minimal example of a core interacting with memory to execute the equivalent of `pimRedSum`. This benchmark should take the same amount of time as `pimRedSum` on a `PimObj` of size `128 * n_banks` 32-bit integers.

```python
# create a core at bank 0
core = Core((0, 0, 0, 0), Ptr(mem))
# define a program
core.add_instruction(OpType.NOP)
core.add_instruction(OpType.READ, operands=[0x0, "reg_vA"])
for i in range(1, int(len(test_list) / 4)):
    core.add_instruction(OpType.READ, operands=[0x10 * i])
    core.add_instruction(OpType.ADD, operands=["reg_vA", 0x10 * i])

core.add_instruction(OpType.ACC, operands=["regA", "reg_vA"])

# simple tick until pipeline is empty
while len(core.instruction_queue) > 0 or not core.pipeline.is_empty():
    core.tick()
    mem.tick()
```

<!-- column: 1 -->

```bash +exec
cd .. && python ./demo/lobsta-wip-red-sum.py
```

<!-- end_slide -->

# Building a Core (WIP)

If we want to explore how adding or removing registers or pipeline stages affects performance, we can do so by changing our definition of core as follows:

```python
core = Core(
    (0, 0, 0, 0),
    Ptr(mem),
    pipeline_stages=["st_fetch", "st_e_exe"],
    registers=["regA"],
    vec_registers=["reg_vA"],
)
```

Notice how we see 2 cycles less time. This is because we have 1 fewer stage!

```bash +exec
cd .. && python ./demo/lobsta-wip-red-sum-less-stages.py
```

<!-- end_slide -->

# Scaling It Up

Below, we simulate the runtime of reduction sum on an input vector of size 65536 (comparing functional output).
```bash +exec
cd .. && python ./demo/lobsta-wip-red-sum-hbm.py
```

You may notice that this figure (1512.0 ns) is approximately 4x larger than the figure obtained from an identical PIMeval execution. This happens for a few reasons:
- PIMeval uses dramatically different row read and write timings, still parsing this
    - PIMeval spends significantly less time performing read operations (not sure how this works yet)
- Pipeline stage adds 1-2 cycles of latency to every read in its current state 
    - After reviewing, it seems I may have fixed this issue already
    - This can be resolved by giving the core at least 2x clock rate relative to RAM
    - There may be other solutions, but have not had time to test them

<!-- end_slide -->

# Making Internal State Legible

Using the below snippet, we can preview the core state every cycle. The output previews said state every 140 cycles.

```python
print("pipeline:", core.pipeline)
print("regA:", core.regA)
print("reg_vA:", core.reg_vA)
```

```bash +exec
cd .. && python ./demo/lobsta-wip-observation.py
```

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
2. Establish instruction standard       \[󱗼\]
3. Replicate LoBSTA Core                \[󱗽\] <- missing scratchpad simulation (easy)
4. Add standardized core infrastructure \[󱗾\]
5. Create UPMEM core/model              \[ \]
6. ...

