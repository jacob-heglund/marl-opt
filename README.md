# MARL with High-Level Model

## Recreating the Results

### Environment Setup

Set up basic docker environment

```bash
cd docker
bash build.sh
```

get GurobiPy version 9.5

- this comes with its own included mini-license (for solving small optimization problems), so you might not need to deal with the whole "get a license and install Gurobi" thing

https://www.gurobi.com/documentation/9.5/quickstart_linux/cs_using_pip_to_install_gr.html#subsubsection:pip