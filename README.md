# Energy Sharing Starter Kit ⚡

An open project repository for energy communities: simulation code for energy sharing, and real-world templates for contracts and governance.

## About
This repository supports energy communities that want to design, test, and implement sharing energy locally. 
It brings together two things:
-	Modular simulation code to explore policy, pricing, allocation, and behavioral choices before deploying them in the real world
-	Real-world artifacts—contract templates, governance documents, and budgets—based on what we actually used in practice

The core philosophy is modularity and reuse: communities should be able to adapt pricing models, allocation rules, and organizational choices without rebuilding everything from scratch. The simulation framework is explicitly designed to make those choices visible, comparable, and debatable—technically, socially, and politically.

## Simulation

The `simulation/` directory contains the core energy sharing simulation. It takes meter data from households and local renewable production sources, and models what happens when they share that energy before drawing from the grid — who gets what, at what cost, and how self-sufficient the community is.

It runs through your data timestep by timestep and produces per-prosumer results: energy allocated, residual grid usage, and costs — both under local sharing and as a counterfactual (what each member would have paid without it).

**Start here:** [`simulation/README.md`](simulation/README.md)  

## Licensing

This repository uses **multiple licenses**, which can be found in the `LICENSES` directory:

- **Software (simulation code)**: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).  
  Applies to all source code under `simulation/` and other code unless stated otherwise.

- **Documentation, business models, and contract templates**: Creative Commons Attribution 4.0 International (CC BY 4.0).  
  Applies to the contents of `simulation/docs/`, `business-models/`, and `contracts/` unless stated otherwise.

Where files include an SPDX license header or a directory contains a license notice README, that notice clarifies the license for that file or directory.
