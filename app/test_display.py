"""Test marimo display."""

import marimo

app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    return (mo,)


@app.cell
def test_header(mo):
    return mo.md("# Test Header - You should see this!")


@app.cell
def test_callout(mo):
    return mo.callout(mo.md("This is a callout"), kind="info")


@app.cell
def test_button(mo):
    btn = mo.ui.button(label="Click me")
    return mo.vstack([
        mo.md("## Controls"),
        btn,
        mo.md(f"Button clicked: {btn.value}"),
    ])


@app.cell
def test_footer(mo):
    return mo.md("---\n*Footer - End of test*")


if __name__ == "__main__":
    app.run()
