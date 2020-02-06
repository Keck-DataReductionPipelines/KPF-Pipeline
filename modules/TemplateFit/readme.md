# Radial Velocity via Template Fitting

This is the implementation of the template fitting method for calculating radial velocity, based on Anglada-Escude and Butler (2012). 

## Module Structure

This module is meant to be under the KPF-Pipeline package. Besides the dependencies listed in `requirement.txt`, this module also depends on the `KeckDRPFramework` package. Since this module is part of the entire KPF pipeline, it will interact with top level folders in many ways. The complete repo looks like:

```txt
KPF-Pipe
├── modules
│   ├── tfa (this module)
|   |   ├── data
|   |   |   └── ... (input .fits files)
|   |   ├── reference
|   |   |   └── ... (useful references)
|   |   ├── test
|   |   |   └── ... (unit tests. working on it)
|   |   ├── tools
|   |   |   └── ... (helpful scrips)
|   |   ├── alg.py
|   |   ├── arg.py
|   |   ├── primitives.py
|   |   └── readme.md
|   └── ... (other non-related modules)
├── makefile (for repo management)
├── requirement.txt (contains dependencies of entire pipeline, including this module
├── logs
|   └── ... (.log files go here)
├── config
|   └── ... (.cfg configs go here)
├── sample_run
|   └── ... (example recipes go here)
├── receipts
|   └── ... (any receipts go here)
└── ... (other non-related folders an files in KPF pipeline)
```


## Using the Module

Work in progress. Some goals are

1. running through KPF pipeline (with recipe and config files)

2. running in debug mode (for improving the algorithm)

3. using some basic tools (like plotting, comparing to CCF, etc.)

4. unit testing 

## Data

Currently the module can only process HARPS data in .fits files. The original HARPS Barnard's Star data are provided as benchmark for testing the modules. Note that the module currently does not support KPF data structures (`KPF1` and `KPF2`) as it should. I find that it is quite complicated to support them when no real data in those structure exist yet, so I am making do with HARPS data for now. Eventually I will try to convert some of the HARPS data into KPF data.

The module outputs a .csv file that contains the final radial velocities from each file, as well as estimated uncertainty. If ran in debug mode (see [Operation](#Operation) for details), the module will also output a folder of debugging details, which also contains .xlsx and .dat files. Running `make clean` will purge these data.

Certain tools will also output .png image files. Again, `make clean` will purge these too.

## Operation

NOTE: details are still being worked out as implementation goes

To operate this module, you must have a .cfg configuration file available as input. The config file must be recognizable by python's `ConfigParser`. An example is provided in the config folder on top level. 

There are two possible ways to run the module: the quick **normal** mode for regular data process, and a much more time and computation intensive **debug** for when you want to see all the details when you're trying to improve algorithm performance. Toggle between the two in the config file. Note that **debug** mode gets its own section in the config file. This section can be complete jibberish when running in **normal** mode, but you must provide valid entries if you want to run in debug mode.

The logger adheres to the standard python `logging` convention. A .log file is generated during each run, and all settings relevant to the logger is specified under the **logging** section of the config files. Typically, an entry in the log file correspond to a non-trivial event in the module (like when a file is processed or a error has occured). Each entry also has a corresponding level. I am using the default `logging` levels: 

| Level    | Usage           |
| -------- |:----------------|
| DEBUG    | For nitty-gritty stuff useful for catching bugs |
| INFO     | For signalling beginning or end of a process      |
| WARNING  | For when something unexpected happened      |
| ERROR    | For errors that the module does not recognize       |
| CRITICAL | For serious errors that may stop the pipeline from continuing       |

One can set the level of the logger in the config file. By default the log entries are also printed to terminal `stdout`. However, if one sets in the configfile `Verbose = False`, then only CRITICAL logs are printed.

## Submodules

This module is broken into four submodules (python files within this folder):

1. alg.py: This file contains the bulk of the template fitting algorithm. All the math and science go here.

2. arg.py: This file defines the data structures used by the template fitting algorithm. They are imported into the alg.py and primitives.py

3. macro.py: This file describes a few data types, constants, and other invariants as global variables.  It also store some general-purpose helper funtions.it does not depend on any other submodules, but is imported by all others.

     It is not recommended to store parameters in this file (global variables in python are bad practice), unless they are used across submodules or just invariants that never are never modified.

4. primitives.py: This file contains things that should happen to the data before or after the actual algorithm is run. They are not officially part of the algorithm, however, they are imported by alg.py.

