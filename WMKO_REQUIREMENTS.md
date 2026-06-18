# KPF DRP — WMKO Technical Requirements (mirror)

> Transcribed from `WMKO_REQUIREMENTS.pdf` — *Keck Planet Finder Data Reduction Pipeline,
> Technical Requirements* (W. M. Keck Observatory, Scientific Software Engineering),
> revision 0.3 (2026-03-16). The PDF is the authoritative source; this Markdown mirror exists
> so the requirements are greppable and linkable alongside the code. Re-transcribe if the PDF
> is revised. Requirement text is reproduced faithfully (including the source's wording).

**Authority precedence (highest first).** The WMKO technical requirements are the **top** of
the project's reference hierarchy — they outrank every other governing document:

**1. these WMKO requirements → 2. the EPRV data standard
([`EPRV_DATA_STANDARD.md`](EPRV_DATA_STANDARD.md)) → 3. the project charter
([`KPF_DRP_VNEXT_CHARTER.md`](KPF_DRP_VNEXT_CHARTER.md)) → 4. the style guide
([`KPF_DRP_VNEXT_STYLE_GUIDE.md`](KPF_DRP_VNEXT_STYLE_GUIDE.md)).**

When any two conflict, the higher one wins.

**Active vs. passive compliance.** The pipeline is in active development and **most of these
requirements are not yet met.** Only flag **active** violations — *existing code that
contradicts a requirement* — not **passive** ones, where a requirement is simply unmet
because the relevant feature has not been built yet. A missing capability is expected and is
**not** a violation; code that does the wrong thing is. Do not add "not implemented" warnings
or block work merely because an as-yet-unbuilt requirement is unmet.

These requirements are adapted from version 1.0 of the Data Processing and Management ICD.
They serve as the core technical requirements for the new Keck Planet Finder (KPF) Data
Reduction Pipeline (DRP) currently in development by the KPF Team.

## Document provenance

| Rev | Description | Date | Who |
|---|---|---|---|
| 0.1 | Initial creation of document | 2026-03-04 | Max Brodheim, Jeff Mader |
| 0.2 | SSG draft review | 2026-03-05 | Max Brodheim, Jeff Mader, Lucas Furhman |
| 0.3 | Initial KPF Team Review | 2026-03-16 | Max Brodheim, Josh Walawender, Andrew Howard, BJ Fulton, Sam Halverson |

**Approvals:** John O'Meara (WMKO Chief Scientist); Marc Kassis (WMKO Instrument Program
Manager); Josh Walawender (KPF Instrument Scientist).

---

## Development Requirements

| ID | Requirement |
|---|---|
| DRP-DEV-01 | The DRP shall be developed using a version of Python between 3.12 and the latest stable release. |
| DRP-DEV-02 | Any use of any non-Python executable code, including C/C++ and shell scripting languages, shall require prior approval by the KPF Instrument Scientist. |
| DRP-DEV-03 | Code revision shall be deployed in GitHub. |
| DRP-DEV-04 | Delivery of the DRP shall consist of transferring the repository to the WMKO organizational repository in GitHub (<https://github.com/Keck-DataReductionPipelines>). |
| DRP-DEV-05 | The DRP must comply with NASA SMD's Open Science guidelines as described in SPD-41a. |
| DRP-DEV-06 | The code shall follow PEP8 Style Guidelines for Python Code. Stricter style guidelines are permitted so long as they comply with PEP8. |
| DRP-DEV-07 | Automatable tests sufficient to demonstrate the operability and basic scientific validity of the DRP shall be provided. |
| DRP-DEV-08 | Each reduction mode(s) shall have a corresponding pre-selected test suite that is publicly available. |
| DRP-DEV-09 | The DRP shall provide documentation covering requirements, installation, description of algorithms and reduction flow, command line interfaces, customization methods, data products, and "cookbooks" using the files identified in DRP-DEV-08. The source code shall contain comments and docstrings describing code flow and intent. Additionally, common errors/failure modes shall be documented, with strategies to manipulate the input data to avoid those errors. |
| DRP-DEV-10 | The code shall be distributed using one of the standard open software licenses (BSD 3-Clause is preferred). |
| DRP-DEV-11 | The DRP shall run on Ubuntu 22.04 LTS or Ubuntu 24.04 LTS, or the latest Ubuntu LTS release. |

## Installation and Build Requirements

| ID | Requirement |
|---|---|
| DRP-BLD-01 | The build and deployment system, defined as packaging tools, installers, dependency resolvers, containerization systems, and/or any other tools and frameworks used in the installation process, shall be approved by the WMKO Scientific Software Group. |
| DRP-BLD-02 | Any database or other system component needed to run the DRP on a user machine shall automatically be created and populated as a part of DRP installation. This excludes any observatory or archive hosted database that are required for pipeline execution. |
| DRP-BLD-03 | The DRP shall create default output directories upon installation, or create those directories upon the first invocation of the pipeline. |
| DRP-BLD-04 | The DRP shall be able to run immediately after a "git clone," fetching of static files, and build. |
| DRP-BLD-05 | Upon installation, the DRP shall use reasonable defaults for all configuration such that the DRP can immediately be run. |
| DRP-BLD-06 | The DRP shall not use environment variables to configure the operation of the pipeline. All pipeline configuration shall be derived from configuration files or command line arguments used upon invocation of a script. Only environment configuration, credentials, and other machine-specific information will use environment variables. |
| DRP-BLD-07 | All environment variables required by the pipeline shall be defined in a `.env` file within the GitHub Repository, populated with example values. |
| DRP-BLD-08 | All static files needed by the pipeline shall be downloaded automatically upon pipeline installation. WMKO can provide file serving as needed. |

## Runtime Requirements

| ID | Requirement |
|---|---|
| DRP-RUN-01 | The DRP shall provide quick-look data products (defined as any output data that an observer might use to determine the scientific viability of a given observation) within 10 minutes, with a goal of <1 minute, to allow modifications to observations in near-real-time. [1] |
| DRP-RUN-02 | "Science level" reductions (i.e. lev2) shall not interfere with the functioning of the quicklook pipeline, nor shall the quicklook pipeline interfere with the functioning of the "Science level" reductions. |
| DRP-RUN-03 | The DRP shall accept files as input from any location on disk as specified by the user at runtime. |
| DRP-RUN-04 | The DRP shall write files to any location on disk as specified by the user at runtime. |
| DRP-RUN-05 | Any files written to disk that are generated from multiple inputs (e.g. "master" calibrations) shall have a filename composed of the KOAID of the first file used, followed by an underscore, followed by a string identifying the type of file. For example, if 3 files (with KOAID's "A" "B" and "C") are combined into a master flat, the output file could be "A_mflat.fits". The suffix string may contain any characters so long as they do not interfere with filepath parsing. |
| DRP-RUN-06 | While running in automatic mode, the DRP shall write data products to disk with the minimum required permissions to facilitate file creation. |
| DRP-RUN-07 | The DRP location to which logs are written shall be configurable by the user. |
| DRP-RUN-08 | DRP logs shall record all reduction steps taken, decision points, occurrences of file reads and writes, warnings, and errors. Warning and error logs shall directly identify what within the pipeline caused the warning or error to occur, and what input files are related. |
| DRP-RUN-09 | All DRP logs shall be saved to the same parent directory. |
| DRP-RUN-10 | The DRP shall record all processing steps applied to each file within the FITS headers of that file, or a dedicated FITS extension. |
| DRP-RUN-11 | The DRP shall write a `DRPVERNO` header keyword that contains the exact version of the pipeline that reduced the data. |
| DRP-RUN-12 | No running instance of the DRP shall prevent another instance of the DRP from operating on different inputs, while on the same server. |
| DRP-RUN-13 | The DRP shall provide a single "one shot" script that takes in a directory and generates all science products possible. |
| DRP-RUN-14 | The DRP shall provide a "real-time" script that starts the pipeline in continuous mode, and can run continuously without user input for a minimum of 24 hours. |
| DRP-RUN-15 | The DRP shall provide a "reprocessing" script that allows for the reduction of more than one directory/date of data in one invocation. |
| DRP-RUN-16 | All scripts used to initiate the DRP shall provide a safe means to terminate that script without leaving the system in an unsafe or unstable state, and allows resumption of execution with another invocation. |
| DRP-RUN-17 | The DRP shall run within a known memory envelope, specified in gigabytes of RAM per file being processed. |
| DRP-RUN-18 | The total memory and CPU resources used by the DRP shall, once steady state processing is reached, not grow over time (i.e. the pipeline shall have no memory or CPU "leaks"). |
| DRP-RUN-19 | The DRP shall propagate the `PROGID` and `KOAID` FITS header keyword through to all FITS files written to disk. |
| DRP-RUN-20 | The DRP shall add a FITS header called `DRPSTATU` that indicates whether a file has been fully reduced. |

## Archive Requirements

| ID | Requirement |
|---|---|
| DRP-KOA-01 | The Keck Instrument Scientist and KOA Scientist shall decide which data products are to be archived in KOA. A summary of these data products shall be presented at the readiness review. |
| DRP-KOA-02 | The DRP shall include a file called `file_io_hooks.py` that will be overwritten by WMKO for deployments at Keck. [2] |
| DRP-KOA-03 | The `file_io_hooks.py` file shall include the following function to be called on file writes [2]: `def file_write_hook(koaid: str, filepath: str, start_time: str, data_level: str = 'lev1')` |
| DRP-KOA-04 | Any function that writes any file, FITS or otherwise, intended for inspection by a typical DRP user (i.e. a fully reduced science output) shall accept as an argument a Boolean value called "final". This "final" value may be a replication of the `DRPSTATU` keyword. |
| DRP-KOA-05 | Any function that writes a file intended for inspection by a typical DRP user shall invoke the `file_write_hook` function when the "final" value is set to true, filling in each argument with the appropriate information. `start_time` is defined as the UTC time at which reduction for that file began and shall use ISO 8601 formatting. |
| DRP-KOA-06 | The DRP shall, through the "final" flag, identify when a file is completely reduced and ready for inspection by an end user or ingestion into the Keck Observatory Archive. |

---

## Footnotes

**[1]** Examples of these quicklook products might include, but are not limited to:

- Basic image processed raw frames (as a FITS file or other image format)
- Plot indicating presence/prevalence of saturated pixels in a frame
- Plot showing guiding errors over the course of an exposure

**[2]** The DRP shall include a file called `file_io_hooks.py` with the following contents:

```python
def file_write_hook(koaid: str, filepath: str, start_time: str,
                    data_level='lev1'):
    pass
```

This function will be overwritten by WMKO for deployments at Keck and will contain all
functionality needed to integrate the DRP with Real Time Ingestion (RTI). It may additionally
contain other functionality written by WMKO to monitor the outputs of the DRP as they are
written. The intention of this requirement(s) is to decouple the development of the DRP and
the archiving infrastructure at WMKO.

The intent is that for non-WMKO users, this invocation will silently pass and not impact their
use of the DRP.
