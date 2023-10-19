# Datasets

This directory contains two main directories of task datasets all in `.json` format.
* (1) The `abstractive` directory contains tasks which require information that is not present in the prompt to answer.
* (2) The `extractive` directory contains tasks where the answer is present somewhere in the prompt, and the task of the model
is to retrieve it.

The `generate` directory contains scripts we used to filter existing datasets, as well as a notebook we used to create new datasets, in addition to cleaning and filter additional pre-existing datasets.