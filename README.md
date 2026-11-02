<img src="kairo/kairo.png" alt="kairo icon" width="96">


# Quickstart


The dashboard runs on [kairo](kairo), the library of this thesis. With [uv](https://docs.astral.sh/uv/) installed, run from the repository root:

```bash
uv sync
uv run streamlit run dashboard/app.py
```

## Docker

Build and run from the repository root (Docker with BuildKit is required):

```bash
docker build -t kairo-dashboard .
docker run --rm --name kairo-dashboard -p 127.0.0.1:8501:8501 kairo-dashboard
```

Open [localhost:8501](http://localhost:8501) and upload an XES or CSV event log.
The image uses Python 3.12, installs the dependencies from `uv.lock` without the
development group, and runs as an unprivileged user. Chromium is included for
Copilot's plot attachments. The first build downloads the full project environment,
including PyTorch and the notebook dependencies; subsequent builds reuse the
dependency cache.

For Copilot, optionally pass API keys at runtime by adding `--env-file .env` before
the image name in `docker run`. The supported variables are `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY`; keys can also be entered in the dashboard.
For a custom LLM server on the Docker host, use `http://host.docker.internal:1234/v1`
in Copilot instead of `127.0.0.1`. On Linux, also add
`--add-host=host.docker.internal:host-gateway` to `docker run`; the LLM server must
listen on an interface reachable from the container.
Local `.env` files, Streamlit secrets, datasets, and thesis files are excluded from
the build context. Uploaded logs and analysis results live in the browser session's
server-side memory and are lost when that session or container ends.

Docker checks `http://127.0.0.1:8501/_stcore/health` inside the container. From another
terminal, check readiness and stop the application with:

```bash
docker inspect --format '{{.State.Health.Status}}' kairo-dashboard
curl --fail http://localhost:8501/_stcore/health
docker stop kairo-dashboard
```

Run the image's application smoke check without API credentials or external data:

```bash
docker run --rm -i kairo-dashboard python - < scripts/smoke_docker.py
```

This checks CSV/XES loading, the dashboard pages, all three feature/PCA/SOM/drift
pipelines, and Copilot's PNG plot generation. It uses Streamlit's test harness with
synthetic session data; it does not send requests to an LLM provider.

The kairo docs, with the approach, a walkthrough and the full API, are at [boraderen.github.io/thesis](https://boraderen.github.io/thesis/) and in [kairo/index.html](kairo/index.html).

# Thesis Proposal

It is highly ML focused (see the reference paper https://www.alessandroberti.it/new_papers/2025_Berti_States_SOM.pdf in which feature extraction+PCA+SOM have been applied) and at the same time concrete enough for a BSc thesis (as you have also to implement a dashboard exploring the states).

## Title

State Based Process Monitoring in Traditional Event Logs for Concept Drift Detection

## Overview

In this thesis, you will study how to describe the behavior of a process by means of several complementary notions of state in a traditional event log. The log is assumed to follow the standard process mining setting, where each event belongs to exactly one case and contains at least a case identifier, an activity, a timestamp, and possibly resource related attributes.

The thesis is inspired by the idea of execution states and boundary conditions from the paper Identifying Execution States and Boundary Conditions in OCELs, but the goal here is not to work with object centric event logs. Instead, you will transfer and adapt the state based perspective to classical event logs.

## Three Types of State

The main goal is to define and study three types of state.

### Intra-Case State

An intra case state should describe the current situation of a single running case. This state should be derived from the sequence of events that has happened in that case so far. Possible ingredients are the executed activities, their order, the time since the previous event, the total time elapsed in the case, and relevant case attributes.

### Resource State

For each relevant resource attribute, you should define a resource state over the whole event log. This state should summarize how the corresponding resource dimension currently behaves across all cases. Examples are workload, handover patterns, waiting times, or the distribution of activities performed by a certain resource, role, or team.

### Inter-Case State

You should define an inter case state that captures the global process situation across cases. This state should be based on the sequence of events in the log as a whole and on temporal gaps between events. The aim is to capture whether the process is currently in a phase such as normal flow, congestion, bursty activity, or unusually slow behavior.

## Core Challenge

A central challenge of the thesis is to make these state notions both computable and interpretable. The states should not only be detected, but also explained. For each type of state, the analysis should answer questions such as: what is characteristic for this state compared with the others, which conditions are typically observed before entering this state, and which conditions are associated with leaving it.

## Implementation Goal

The implementation goal is to build a dashboard that supports the exploration of these states. After the states have been computed, the dashboard should allow the user to inspect each state, compare it with other states, and understand its main characteristics. It should also visualize the conditions that often lead to entering or exiting the state. In addition, the dashboard should show how frequently each state occurs over time, so that the evolution of the process can be monitored.

## Evaluation Goal

The evaluation goal is to investigate whether the combination of these states can serve as a useful signal for concept drift detection. In other words, you will study whether significant changes in the intra case, resource, and inter case states indicate that the underlying process has changed. This can be evaluated by transforming the states and their frequencies into time dependent signals and checking whether changes in these signals correspond to known or injected drift points.

## Instance-Spanning Constraints

A further aspect of the thesis is the connection to instance spanning constraints. These constraints refer to dependencies that involve multiple cases rather than a single case only. In this thesis, they can be used in two ways. First, they can help characterize inter case and resource related states. Second, they can support the interpretation of state changes by showing whether a shift in state is linked to cross case interactions, competition for shared resources, or other global effects.

## Research Question

Can a process be monitored through a combination of intra case, resource, and inter case states, and is this combination a useful indicator of concept drift in traditional event logs?

## Work Plan

1. Study (not too deep) the literature on state detection, concept drift detection, and instance spanning constraints in process mining.
2. Adapt the notion of execution state from the inspiration paper to the setting of traditional event logs.
3. Define features for intra case, resource, and inter case states.
4. Implement a method to compute these states from an event log.
5. Design and implement a dashboard for exploring the states and their transitions.
6. Evaluate whether state based signals can detect concept drift.
