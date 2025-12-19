"""benchmORk - Interactive Benchmarking Dashboard."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return mo, pd, alt


@app.cell
def load_benchmork(mo):
    try:
        from benchmork.config import get_config
        from benchmork.problems import TransportationProblem, DietProblem
        from benchmork.runner import run_benchmark
        from solvers import get_available_solvers, get_solver_by_name, SOLVER_REGISTRY

        config = get_config()
        load_error = None

        status_msg = mo.callout(
            mo.md("**Ready!** Select a problem type to see available solvers."),
            kind="success",
        )
    except ImportError as e:
        status_msg = mo.callout(
            mo.md(f"**Error loading benchmork:** {e}"),
            kind="danger",
        )
        config = None
        load_error = str(e)
        TransportationProblem = None
        DietProblem = None
        run_benchmark = None
        get_available_solvers = None
        get_solver_by_name = None
        SOLVER_REGISTRY = {}

    return (
        TransportationProblem,
        DietProblem,
        run_benchmark,
        get_available_solvers,
        get_solver_by_name,
        SOLVER_REGISTRY,
        config,
        status_msg,
        load_error,
    )


@app.cell
def header(mo, status_msg):
    return mo.vstack([
        mo.md("""
        # benchmORk
        ### Optimization Solver Benchmarking Dashboard
        """),
        status_msg,
    ])


@app.cell
def problem_dropdown(mo, config):
    """Problem type dropdown."""
    problem_opts = {"Linear Programming (LP)": "linear", "Mixed Integer Programming (MIP)": "integer"}

    if config and config.problem_types:
        problem_opts = {
            pt.display_name: pt.id
            for pt in config.problem_types.values()
            if pt.id in ("linear", "integer")
        }

    problem_select = mo.ui.dropdown(
        options=problem_opts,
        value=list(problem_opts.keys())[0] if problem_opts else None,
        label="Problem Type",
    )

    return problem_select, problem_opts


@app.cell
def solver_multiselect(mo, problem_select, get_available_solvers):
    """Solver multiselect - filtered by selected problem type."""
    prob_type = problem_select.value

    if get_available_solvers and prob_type:
        solvers_available = get_available_solvers(prob_type)
        # Use solver.name for both display and value (BaseSolver has no display_name)
        solver_opts = {s.name: s.name for s in solvers_available}
    else:
        solvers_available = []
        solver_opts = {}

    solver_select = mo.ui.multiselect(
        options=solver_opts,
        value=list(solver_opts.keys()),
        label="Solvers",
    )

    return solver_select, solver_opts, solvers_available, prob_type


@app.cell
def other_controls(mo):
    """Size slider and repeats input."""
    size_input = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=20,
        label="Problem Size",
        show_value=True,
    )

    repeat_count = mo.ui.number(
        start=1,
        stop=10,
        value=3,
        label="Repeats",
    )

    run_btn = mo.ui.run_button(label="Run Benchmark")

    return size_input, repeat_count, run_btn


@app.cell
def controls_panel(mo, problem_select, solver_select, solver_opts, size_input, repeat_count, run_btn):
    """Display all controls."""
    solver_count = len(solver_opts)
    solver_info = mo.md(f"*{solver_count} solver{'s' if solver_count != 1 else ''} available*")

    return mo.vstack([
        mo.md("## Configuration"),
        mo.hstack([problem_select], justify="start"),
        solver_info,
        mo.hstack([solver_select, size_input, repeat_count], justify="start", gap=2),
        run_btn,
    ])


@app.cell
def run_benchmark_cell(
    mo,
    run_btn,
    prob_type,
    solver_select,
    solver_opts,
    size_input,
    repeat_count,
    TransportationProblem,
    DietProblem,
    run_benchmark,
    get_solver_by_name,
    load_error,
):
    """Execute benchmark when button clicked."""
    mo.stop(not run_btn.value, mo.md("*Click 'Run Benchmark' to start*"))
    mo.stop(load_error is not None, mo.callout(mo.md(f"Cannot run: {load_error}"), kind="danger"))

    selected_names = [solver_opts[name] for name in solver_select.value]
    benchmark_solvers = [get_solver_by_name(n) for n in selected_names]
    benchmark_solvers = [s for s in benchmark_solvers if s is not None]

    mo.stop(not benchmark_solvers, mo.callout(mo.md("No solvers selected"), kind="warn"))

    size = size_input.value

    if prob_type == "linear":
        problem = TransportationProblem(n_sources=size, n_destinations=size)
        prob_label = f"Transportation {size}x{size}"
    else:
        problem = DietProblem(n_foods=size * 2, n_nutrients=max(2, size // 2))
        prob_label = f"Diet {size * 2} foods, {max(2, size // 2)} nutrients"

    with mo.status.spinner(title="Running benchmark..."):
        results = run_benchmark(
            solvers=benchmark_solvers,
            problems=[problem],
            repeats=repeat_count.value,
            warmup=True,
        )

    return results, problem, prob_label


@app.cell
def results_table(mo, results, prob_label):
    """Display results table."""
    mo.stop(results is None, mo.md(""))

    table_df = results.to_dataframe()
    fastest_result = results.get_fastest()

    summary = mo.md(f"""
    ## Results

    **Problem:** {prob_label}

    **Fastest solver:** {fastest_result.solver if fastest_result else 'N/A'} ({fastest_result.solve_time_mean:.4f}s)
    """)

    table = mo.ui.table(
        table_df[[
            "solver", "problem", "status", "objective_value",
            "solve_time_mean", "solve_time_std", "setup_time_mean"
        ]].round(6),
        selection=None,
    )

    return mo.vstack([summary, table])


@app.cell
def results_chart(mo, alt, results):
    """Display timing chart."""
    mo.stop(results is None, mo.md(""))

    chart_df = results.to_dataframe()
    mo.stop(chart_df.empty, mo.md(""))

    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("solver:N", title="Solver"),
            y=alt.Y("solve_time_mean:Q", title="Solve Time (seconds)"),
            color=alt.Color("solver:N", legend=None),
            tooltip=["solver", "solve_time_mean", "solve_time_std", "status"],
        )
        .properties(width=400, height=300, title="Solve Time Comparison")
    )

    error_bars = (
        alt.Chart(chart_df)
        .mark_errorbar()
        .encode(
            x=alt.X("solver:N"),
            y=alt.Y("solve_time_mean:Q"),
            yError=alt.YError("solve_time_std:Q"),
        )
    )

    timing_chart = mo.ui.altair_chart(bars + error_bars)

    return mo.vstack([
        mo.md("## Timing Comparison"),
        timing_chart,
    ])


@app.cell
def footer(mo):
    return mo.md("""
    ---
    *benchmORk* - Compare optimization solvers with ease.
    """)


if __name__ == "__main__":
    app.run()
