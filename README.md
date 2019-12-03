# IMB-Mulan
## The Mulan Framework with Multi-Label Resampling Algorithms

The Imbalanceness Mulan (IMB-Mulan) is an extension to the well-known Mulan framework (https://github.com/tsoumakas/mulan) with implementations of resampling algorithms previously proposed in the literature.

# Resampling Algorithms
The following resampling algorithms were implemented in the IMB-Mulan framework:
- Label Powerset Random Oversampling (LPROS)
- Label Powerset Random Undersampling (LPRUS)
- Multi-Label Random Oversampling (MLROS)
- Multi-Label Random Undersampling (MLROS)
- Best First Oversampling (MLBFO)
- Multi-Label edited Nearest Neighbor (MLeNN)
- Multi-Label Synthetic Minority Oversampling (MLSMOTE)
- Multi-Label Resampling by Decoupling Highly Imbalanced Labels (REMEDIAL)
- Multi-Label Resampling by Decoupling Highly Imbalanced Labels with Hybridization (REMEDIAL-HWR)
- Multi-Label Tomek Link (MLTL)

# Examples of use
The package "mulan.resampling.examples" contains codes samples explaining how to use all resampling algorithms implemented.
The following code gives a brief explanation concerning the LPROS resampling algorithms.

```java
//Creating the original dataset
MultiLabelInstances originalTrainingSet = new MultiLabelInstances(arffFilename, xmlFilename);
//Instantiate the LPROS algorithms
LPROS lpros = new LPROS(originalTrainingSet, xmlFilename);
//Resample the original training set
MultiLabelInstances resampledTrainingSet = lpros.resample();
...
```

# Cite
If you used IMB-Mulan in your research or project, please cite our work:

```bibtex
@article{2020pereiramltl,
   author = {Pereira, R. M and Costa, Y. M. G. and Silla Jr., C. N.},
   title = {MLTL: A multi-label approach for the Tomek Link undersampling algorithm},
   journal = {Neurocomputing},
   year = {2020},
   publisher={Elsevier}
}
```

# Contributing
This project is open for contributions. Here are some of the ways for you to contribute:

- Bug reports/fix
- Features requests
- Use-case demonstrations
- Documentation updates

To make a contribution, just fork this repository, push the changes in your fork, open up an issue, and make a Pull Request!
