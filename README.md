```


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~                                                                                                 ~
~   ____                 ___       __            ______            ____    ______                 ~
~  /\  _`\           __ /\_ \     /\ \          /\  _  \          /\  _`\ /\__  _\   /'\_/`\      ~
~  \ \ \L\ \  __  __/\_\\//\ \    \_\ \         \ \ \L\ \         \ \ \L\ \/_/\ \/  /\      \     ~
~   \ \  _ <'/\ \/\ \/\ \ \ \ \   /'_` \  _______\ \  __ \  _______\ \ ,__/  \ \ \  \ \ \__\ \    ~
~    \ \ \L\ \ \ \_\ \ \ \ \_\ \_/\ \L\ \/\______\\ \ \/\ \/\______\\ \ \/    \_\ \__\ \ \_/\ \   ~
~     \ \____/\ \____/\ \_\/\____\ \___,_\/______/ \ \_\ \_\/______/ \ \_\    /\_____\\ \_\\ \_\  ~
~      \/___/  \/___/  \/_/\/____/\/__,_ /          \/_/\/_/          \/_/    \/_____/ \/_/ \/_/  ~
~                                                                                                 ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

```

# Introduction

Welcome to _Build-A-PIM_, a library for building and iterating on Processing in
Memory (PIM) microarchitectures architectures built upon DRAMsim3.

# Getting Started

Before getting started, please note that we currently only support MacOS and
Linux. If you are on a Windows machine, please use WSL to build and run this
project.

## Cloning

To clone _Build-A-PIM_, run the following command:

```bash
git clone git@github.com:UVA-LavaLab/build-a-pim.git --recurse-submodules # ssh
git clone https://github.com/UVA-LavaLab/build-a-pim.git --recurse-submodules # https
```

## Building

We have included a build script for your convenience, which builds our extension
of DRAMsim3 (required for proper functionality). This means that building is as
simple as running `./build.sh` in your terminal.

From there, you should create a Python virtual environment, activate it, and run
the following to install our Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Test Cases

Before testing out our library, be sure to run the test cases by using the
`pytest` command in your project root.

## Testing Out Build-A-PIM

You can test out _Build-A-PIM_ by writing your own scripts in `./scratch/`. If
you're looking for a guide, feel free to visit our collection of demos.

# Demos

A collection of demo programs can be previewed in `./demo/`. Each demo must be
executed from the root directory of the project (because they use relative
imports).

# Attributions

We would like to extend a special thank you to our advisors, Kevin Skadron and
Sandhya Dwarkadas for providing excellent guidance through the implementation of
this library.

We would also like to give a special thank you to Farzana Ahmed Siddique for
consulting regarding hardware assumptions and behaviors during the
implementation process.

Developers:

- William Bradford
- Nebil Ozer
