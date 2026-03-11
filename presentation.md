---
title: _Build-A-PIM_ --- Extending DRAMSim3 to Support Bank-level Access Latency Measurement for Rapid PIM Design Scaffolding
author: William Bradford
theme:
    name: catppuccin-mocha
---

# What is _Build-A-PIM_?

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

# API

The API for _Build-A-PIM_ is rather difficult to parse, which is why it is
intended for use with its much simpler Python wrapper. You can begin by
creating a memory system as shown below:

```Python
mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".")
mem.toggle_pim_mode()
```

Data structures in high-level languages can be "memory-mapped" as shown below.
Importantly, this does NOT simulate transfer rate or timing. While this feature
might be implemented in the future, it is not currently supported.

```Python
test_list = [1, 2, 3, 4, 5]
#em.add_data_structure(   data  , size)
mem.add_data_structure(test_list,   4 )
mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list)*4, offset=0)
```

The above code maps the entirety of `test_list` to 4-byte data items in bank 1,
starting at address 0x0. The `offset` parameter states that it begins the
mapping at byte 0 of the data structure.

<!-- end_slide -->

# API

Performing a memory operation is as simple as using the following method with a
bank-local address.

```Python
def add_transaction_to_bank(
    self,
    channel: int,
    rank: int,
    bankgroup: int,
    bank: int,
    addr: int,
    is_write: bool,
    is_pim: bool, # specifies if *each* operation is PIM
) -> bool:...
# ex:
mem.add_transaction_to_bank(channel=0, 
                            rank=0,
                            bankgroup=0,
                            bank=3,
                            addr=0x06,
                            is_write=True,
                            is_pim=True)
```

<!-- end_slide -->

# Random Access Demo

The following demo shows a basic use case for one bank's PIM core accessing its
local memory

```Python
for _ in range(4):
    mem.add_transaction_to_bank(channel=0, rank=0,
                                bankgroup=0, bank=0,
                                addr=random.randint(0x0,0x10),
                                is_write=True, is_pim=True)
    mem.tick(until_event=True)
    print("Transaction completed at:", mem.m_cycle,
          "Accessed addr:", hex(mem.m_writes[-1][0]))
```

```bash +exec
python ./demo/random-access.py
```

<!-- end_slide -->

# Using DRAMSim3 Features

Now, because we are using DRAMSim3, we already have an infrastructure for
generating things like the command trace:

```bash +exec
head -n 25 ./dramsim3ch_0cmd.trace
```
<!-- end_slide -->

# Building a Core

In this section, we will build a basic DDR4-based PIM system with two
bank-level processors placed in bankgroup 0 at banks 0 and 1 respectively.
Below, we see an implementation of an "Adder" core which iterates from the
bottom of its address space forward by 16 bytes and sums 32-bit values in that
range. The tick implementation is left to your imagination due to its size.

```Python
class Adder:
    def __init__(self, rank: int, bankgroup: int, bank: int):
        self.rank: int = rank
        self.bank: int = bank
        self.bankgroup: int = bankgroup
        self.is_waiting: bool = False
        self.active_addr: int = 0x0
        self.gdl: Any = [0, 0, 0, 0]
        self.regA: int = 0
        self.ptr: int = -1

    def tick(self, mem: MemSystem): #{implementation here}
```

<!-- end_slide -->

# Building a Device

Then, setting up a device is as simple as shown below:
```Python
class Device:
    def __init__(self, config_file: str):
        self.mem: MemSystem = MemSystem(config_file, ".", nd_log=True)
        self.mem.toggle_pim_mode()
        self.adders: list[Adder] = [Adder(0, 0, 0), Adder(0, 0, 1)]

    def tick(self):
        for a in self.adders:
            a.tick(self.mem)
        self.mem.tick()
```

We can make dummy lists and "memory map" them.

```Python
    def setup(self):
        test_list_a = [1, 2, 3, 4]
        test_list_b = [5, 6, 7, 8]
        self.mem.add_data_structure(test_list_a, 4)
        self.mem.add_data_structure(test_list_b, 4)
        self.mem.mmap(
            0, 0, 0, 0, 0x0, data_index=0, 
            length=len(test_list_a) * 4, offset=0
        )
        self.mem.mmap(
            0, 0, 0, 1, 0x0, data_index=1, 
            length=len(test_list_b) * 4, offset=0
        )
```

<!-- end_slide -->

# Using our Device

The following main method can then be used to simulate our device.

```Python
if __name__ == "__main__":
    device = Device("./dramsim3/configs/DDR4_8Gb_x16_3200.ini")
    device.setup()
    ticks = 0
    while device.adders[0].regA < 10 and device.adders[1].regA < 26:
        ticks += 1
        device.tick()
    print("regA: core 0=", device.adders[0].regA, "core 1=", device.adders[1].regA)
    print("Cycles taken:", ticks)
```

```bash +exec
python ./demo/device-demo.py
```

<!-- end_slide -->

# But what if...?

What if we want to test our **same design** implemented in HBM instead?

```Python
if __name__ == "__main__":
    device = Device("./dramsim3/configs/HBM2_8Gb_x128.ini.ini")
    ...
```

Changing the memory logic is as simple as changing the config file.

```bash +exec
python ./demo/device-demo-hbm.py
```

<!-- end_slide -->

# Testing

We have a variety of tests which cover both
relationships between timings and raw timing details
in both HBM and DDR4, implemented via pytest.

```bash +exec
pytest -q
```

Some parts of the library have not been fully-tested:
- The memory mapping system
- The GDL access system
- Running the original DRAMSim3 tests 
    - this should pass, but has not been run yet

<!-- end_slide -->

# API Updates

<!-- column_layout: [3, 2] -->
<!-- column: 0 -->

Over the break, there have been a few updates to the API.

```Python
    gdl = mem[0, 0, 0, 0, 0]
    mem.tick(until_event=True)
    # safety, but gdl.is_ready MUST be called
    while not gdl.is_ready:
        mem.tick(until_event=True)
    gdl.data[1] = 55
    dsetter = DataSetter(gdl)
    mem[0, 0, 0, 0, 0] = dsetter
    mem.tick(until_event=True)
    while not dsetter.output.is_ready:
        mem.tick(until_event=True)
    gdl = mem[0, 0, 0, 0, 0]
    mem.tick(until_event=True)
    while not gdl.is_ready:
        mem.tick(until_event=True)
```

<!-- column: 1 -->

```bash +exec
python ./demo/demo-data-wrapper.py
```

<!-- end_slide -->

# Going Forward

***Disclaimer:*** This list is **enormous**: you can probably expect 3-4 of these things to be done in the "near-future."

- The API could use some additional fine-tuning
    - It would be nice to have the call to access and the call to load the GDL
    as part of one API call
- Add some more on-the-rails simulation for PIM cores 
    - Maybe a general class that can easily interface with the memory system
    - Reach out to Deyuan regarding area estimations based on processor
    functionality
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
