# Notebooks to reproduce results shown in paper.

(Estimated run-times are shown in brackets.)

1. **analytical.ipynb/.py** (~ 35 min): Figures 1, 2, 3, and 5. Table 1.
2. **filter-comparison.ipynb/.py** (~ 35 min): Figure 4.
3. **gpr-create-data.ipynb/.py** (~ 1 day): Create data for figure 6.
4. **gpr-figure.ipynb/.py** (< 1 min): Figure 6.
5. **runtimes.ipynb** (~ 6 min): Tables 2 and 3.
6. **time-domain.ipynb/.py** (< 1 min): Figure 7 and Table 4.

Run-times were measured on a Lenovo ThinkCentre running Ubuntu 16.04 64-bit,
with 8 GB of memory and an Intel Core i7-4770 CPU @ 3.40GHz x 8.

The reason for the long calculation time of *analytical.ipynb* and
*filter-comparison.ipynb* is the size of the model they calculate: 10.5 by 10.5
km on a regular grid with 10 m spacing, hence over 1 million points for each of
the models. In the GPR-case it is not necessary because of the number of
offsets nor frequencies involved, but because of the slow convergence at these
high frequencies.

The file *printinfo.py* is a small routine to print system- and version-info.

The routines are also provided as pure Python files. However, timing was
carried out with a so-called magic-functions built into IPython (%timeit). The
pure Python files do not have the timing bit. For the same reason there is no
pure Python file of *runtimes.ipynb*.
