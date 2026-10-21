Cycles Renderer
===============

Fork by tammi3 to implement Vulkan as a backend for Cycles X, reference https://github.com/tammi3/Vulkan-backend-Blender-CyclesX .

Currently VulkanDevice initializes but i haven't even began writing the kernel dispatch, still figuring out if the Slang shader should be compiled at runtime or 

if there's a way to have it statically and populate the memory, i swear i hear the Dracula motif every time i'm reminded about the integrator


Cycles is a path tracing renderer focused on interactivity and ease of use, while supporting many production features.

https://www.cycles-renderer.org

## Building

Cycles can be built as a standalone application or a Hydra render delegate. See [BUILDING.md](BUILDING.md) for instructions.

## Examples

The repository contains example xml scenes which could be used for testing.

Example usage:

    ./cycles scene_monkey.xml

You can also use optional parameters (see `./cycles --help`), like:

    ./cycles --samples 100 --output ./image.png scene_monkey.xml

For the OSL scene you need to enable the OSL shading system:

    ./cycles --shadingsys osl scene_osl_stripes.xml

## Contact

For help building or running Cycles, see the channels listed here:

https://www.cycles-renderer.org/development/
Small change made on 2026-10-21 to create a visible diff so the Publish to GitHub button has something to show.
Uncommitted working-tree change so the website UI shows a pending diff.
