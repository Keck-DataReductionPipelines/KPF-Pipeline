# KPF-DRP vNext: Project Context and Intent

> **Authority precedence.** This charter is the single source of truth for project intent
> and design, but it is not the top of the hierarchy. Two documents outrank it:
> **1. the WMKO technical requirements ([`WMKO_REQUIREMENTS.md`](WMKO_REQUIREMENTS.md)) →
> 2. the EPRV data standard ([`RVDATA_STANDARD.md`](RVDATA_STANDARD.md)) → 3. this
> charter → 4. the style guide ([`KPF_DRP_VNEXT_STYLE_GUIDE.md`](KPF_DRP_VNEXT_STYLE_GUIDE.md)).**
> When they conflict, the higher one wins. The WMKO requirements are Keck's binding technical
> requirements for the DRP; most are not yet met because the pipeline is in active development
> — flag only *active* violations (existing code that contradicts a requirement), not *passive*
> ones (a requirement unmet because the feature does not exist yet).

## 1. Why We Are Rebuilding

The current DRP has reached a point where:

-   Development velocity is limited by framework complexity.
-   Debugging is slowed by hidden state and infrastructure coupling.
-   Reprocessing is brittle.
-   Calibration strategy (especially WLS + drift) needs structural
    rethinking.
-   Silent failures and nondeterministic behavior undermine confidence.
-   Long-term RV stability remains the dominant scientific challenge.

This effort is not a cosmetic refactor.

It is a controlled reset aimed at restoring: - scientific confidence, -
deterministic behavior, - and development agility.

------------------------------------------------------------------------

## 2. Immediate Objective (Path 3)

We are beginning with a simple, astronomy-style pipeline.

This means:

-   Explicit Python modules.
-   File-in / file-out execution.
-   Minimal abstraction layers.
-   No workflow framework.
-   No service infrastructure.
-   No database dependency.
-   No orchestration engine.
-   No premature architecture.

The goal is not elegance.

The goal is: - clarity, - reproducibility, - and stability.

We optimize for: - ease of debugging, - transparent calibration logic, -
and rapid iteration.

------------------------------------------------------------------------

## 3. Definition of Success (Initial 2--3 Month Target)

The pipeline is considered successful when:

1.  It runs deterministically.
2.  It can reprocess a frozen truth dataset without random failures.
3.  It supports an independent HCL-based WLS path.
4.  Calibration strategies are explicit and comparable.
5.  RV performance on standard stars is stable and measurable.
6.  Silent failures are eliminated.

We are not targeting: - perfect architecture, - distributed scaling, -
or production-grade orchestration.

------------------------------------------------------------------------

## 4. Scientific Focus

The tallest tent pole is:

Intermediate and long-term RV stability.

Everything in this rebuild should serve one of:

-   Improving wavelength calibration robustness.
-   Improving drift modeling clarity.
-   Eliminating outliers.
-   Making calibration failure modes explicit.
-   Enabling systematic regression testing.

If a feature does not support RV stability or reproducibility, it is
deprioritized.

------------------------------------------------------------------------

## 5. Calibration Philosophy Reset

The previous pipeline implicitly centered LFC as the foundational WLS.

In this rebuild:

-   Calibration paths must be explicit.
-   LFC, HCL, and etalon-based strategies are separate modules.
-   Each path must be independently testable.
-   No calibration source is trusted blindly.
-   QC metrics must accompany every calibration product.

Simplicity is preferred over complexity.

------------------------------------------------------------------------

## 6. Development Model

This is a small-team effort:

-   BJ: project lead / scientific oversight / integration.
-   Greg: primary implementation.

Weekly cadence: - measurable progress, - regression metrics, - explicit
sprint goals.

Every major change must: - preserve deterministic behavior, - run on the
truth dataset, - and document impact on RV metrics.

------------------------------------------------------------------------

## 7. Why Path 3 First

We intentionally begin with Path 3 because:

-   Framework friction currently slows iteration.
-   Architecture debates are distracting.
-   Scientific stability must precede software polish.

Path 3 gives us: - agility, - clarity, - and control.

------------------------------------------------------------------------

## 8. Eventual Migration to Path 2

This codebase should evolve toward a cleaner, modern architecture once:

-   Calibration logic stabilizes.
-   RV performance is understood.
-   Silent failures are eliminated.
-   Reprocessing is robust.

Path 2 (modern orchestration, containerization, workflow engines,
unified DB) is a second step, not the first.

We do not prematurely optimize infrastructure.

------------------------------------------------------------------------

## 9. Guardrails

We must avoid:

-   Rebuilding a hidden framework unintentionally.
-   Adding implicit global state.
-   Embedding database dependencies in science code.
-   Introducing silent retries.
-   Overengineering abstractions.

All steps must remain: - explicit, - readable, - debuggable.

------------------------------------------------------------------------

## 10. Core Design Principles

1.  No hidden state.
2.  No implicit calibration assumptions.
3.  Deterministic stacking and WLS.
4.  Fail loudly.
5.  Log everything relevant to calibration and RV derivation.
6.  Write QC metrics alongside products.
7.  Prefer clarity over cleverness.
8.  Implement everything in the simplest possible way.

------------------------------------------------------------------------

## 11. Long-Term Vision

The end state is:

A stable, validated KPF DRP that:

-   Produces publication-ready RVs.
-   Can be reprocessed uniformly across eras.
-   Has explicit calibration fallbacks.
-   Supports future orchestration wrapping.
-   Is testable and maintainable.

The current rebuild is the foundation of that.
