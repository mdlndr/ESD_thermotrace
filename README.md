# **ESD_thermotrace v1.1**

Madella, Andrea; Glotzbach, Christoph; Ehlers, Todd (2021): ESD_thermotrace, A new software to interpret tracer thermochronometry datasets and quantify related confidence levels. V. 1.1. GFZ Data Services. https://doi.org/10.5880/fidgeo.2021.003


# **What is ESD_thermotrace?**

A [jupyter notebook](https://jupyter.org/) that helps interpreting detrital tracer thermochronometry datasets and quantifying the statistical confidence of such analysis. it has been developed by A. Madella in the [Earth Surface Dynamics group of the University of Tübingen](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/geowissenschaften/arbeitsgruppen/mineralogie-geodynamik/forschungsbereich/geologie/earth-system-dynamics-research-group/).
This is version 1.1 of the code and it was last updated in February 2022.


# **Ok, great, but what *exactly* does it do?**

The files in this folder having *.ipynb* and *.py* extensions host all the code. There, a detailed description of all the new routines can be found, as well as references to the used Python libraries. In addition, it is recommended reading the manuscript where ESD_thermotrace is presented and described in better context and detail.
The workflow of ESD_thermotrace is briefly summarized here below. For each main step, we specify inputs, outputs and methods.

**1) Bedrock age map interpolation**

*Input*

- Bedrock age dataset (table data)
- Digital elevation model of the studied catchment (grid data)
- Cellsize (user-defined resolution of the age map)

*Output*

- Bedrock age map (grid data)

*Method*

In this step a bedrock age map is computed, such that differences to the observed dataset are minimized. Here, users can choose among the following methods:
- [1D inverse variance weightes linear regression](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html) (bedrock age variance is explained only by changes in elevation)
- [3D linear Radial Basis Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html#scipy.interpolate.Rbf) (bedrock age variance is explained by changes in X, Y, Z coordinates)
- [3D linear interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata) (bedrock age variance is explained by changes in X, Y, Z coordinates)
Alternatively, users can import their own age and age-uncertainty map.

**2) Bedrock age uncertainty map interpolation**

*Input*

- Bedrock age dataset (table)
- Bedrock age map (grid data)

*Output*

- Bedrock age uncertainty map

*Method*

The uncertainty of the age map is estimated through bootstrapping. This means that the age map is iteratively recalculated as many times as the measured bedrock ages. For each iteration i, one age A<sub>i</sub> is excluded from the input. The difference between the excluded age A<sub>i</sub> and the interpolated age is taken as the age error E<sub>i</sub> at the coordinates X<sub>A<sub>i</sub></sub>, Y<sub>A<sub>i</sub></sub>. Where the age analytical error is larger than the interpolation error, E<sub>i</sub> will equal the root of the sum of the squared errors. The age uncertainty map is then interpolated from all the local errors E<sub>i</sub>.
Alternatively, If users opted to import their own age map, a related age uncertainty map is required too.

**3) Extract catchment bedrock age, coordinates and erosion data**

*Input*

- Outline of the catchment (shapefile)
- Bedrock age and age uncertainty maps (grids)
- Mineral fertility map (grid, optional)
- Erosion scenarios (grid data and/or functions of X,Y,Z written in Python)

*Output*

A table of catchment data with one row per catchment cell and the following columns:
- coordinates X, Y, Z
- age and age uncertainty
- Mineral fertility
- one column per erosion scenario, each informing the local erosional weight.

*Method*

The data listed just above is saved into an excel table for each cell that falls within the imported catchment outline. A column for uniform erosion scenario "Euni" is included by default.

**4) Predict detrital grain age distributions for each erosion scenario**

*Input*

- Extracted table of catchment data

*Output*

- A predicted detrital age population per erosion hypotheses, having n>>1000
- A predicted detrital age distribution for each of the erosion hypotheses

*Method*

Age populations are predicted as follows: for each scenario, a number of ages is drawn from a normal distribution in each cell. This normal age distribution is constructed from the local mean age and age uncertainty. The drawn number of grain ages is the product of the local mineral fertility, the erosional weight and an arbitrary multiplier (user-defined and constant for all cells).  For each predicted population, ages are sorted to construct the related cumulative age distribution.

**5) (a) Evaluate the confidence of rejecting the uniform erosion scenario (*Euni*) based on the available grain-ages; (b) study the statistical power of discerning each imported erosion scenario from *Euni*.**

*Input*

- One or more sets of measured detrital grain ages and related analytical uncertainty (table)
- Predicted detrital populations and distributions (stored in a Python dictionary)

*Output*

- A graph informing the confidence level allowed by the sample size as well as the statistical power of discerning between erosion scenarios and *Euni*, as a function of the number of observed grain-ages (an editable .pdf file)

*Method*

First, the least significant dissimilarity *Dcrit* for the available sample size *k* is calculated analytically with Equation 3 of the manuscript. Then, a large number of n=*k* distributions are drawn from the pool of observed grain-ages (applying the mean analytical uncertainty to all ages). The fraction of distributions that is more dissimilar to *Euni* than *Dcrit* represents the confidence level allowed by the sample size.
The same operation is also repeated for a range of sample sizes (20<*k*<130), in order to estimate how the confidence level would benefit from an increase in sample size (if the observed distribution and associated uncertainty remained identical despite the changing sample size).
To calculate the statistical power of discerning the tested erosion hypotheses, the same approach is applied. In this case, however, the software draws a number of distributions from each erosion scenario, instead of from the observed grain-ages.

**6) Evaluate which of the erosion scenarios produces distributions that are least dissimilar to the observed detrital distribution**

*Input*

- One or more sets of measured detrital grain ages and related analytical uncertainty (table)
- Predicted detrital populations and distributions (stored in a Python dictionary)

*Output*

- A violin plot informing the distribution of dissimilarities between the erosion scenarios and the observed data, as well as the "plausibility" of each scenario (editable .pdf file)
- A Multidimensional scaling (MDS) plot, showing the degree of overlap among the scenario distributions and the observed detrital age distribution (editable .pdf file)

*Method*

The dissimilarity between the observed cumulative age distribution (from *k* grain-ages) and a n=*k* subset of each predicted detrital distribution is calculated 10000 times. Then, the distribution of these dissimilarities are plotted in the form of a violin plot. The "plausibility" of each scenario represents the likelihood that they produce distributions that cannot be statistically discerned from the observed detrital distribution, with the available sample size.
The MDS plot is constructed following the approach described by [Vermeesch (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0009254113000387)


# **Installation and quick usage guide**

Perhaps you already have a Python 3 installation with jupyter,
but ESD_thermotrace might use some libraries you still have not installed.
To ensure a painless installation and usage, this is the way to go:

Make sure you have an Anaconda installation with Python 3. If not, [follow these instructions](https://docs.anaconda.com/anaconda/install/)

Then Clone or download this repository to your preferred directory.

Open a terminal (MacOS, Linux) or Anaconda-Prompt window (Windows).

Go to the downloaded *esd_thermotrace* directory.

Create a new environment from the provided .yml file by entering the following command:
```
conda env create -f ESD_thermotrace_1_environment.yml
```
Activate the environment like so
```
source activate ESD_thermotrace
```
or (depending on Anaconda version and operating system)
```
conda activate ESD_thermotrace
```
Launch jupyter
```
jupyter notebook
```
Open the notebook *ESD_thermotrace_1.ipynb* in the browser window, you're good to go!

Press **SHIFT+ENTER** to run the current cell

To close the program, enter CTRL+C in the terminal, or click the Quit button in the home page of the User Interface.

Please refer to the [jupyter documentation](https://jupyter-notebook.readthedocs.io/en/stable/) for all other questions on how to use notebooks.

To return to the base Anaconda environment, enter the activation command without specifying the env. name:
```
source activate
```
or
```
conda activate
```
