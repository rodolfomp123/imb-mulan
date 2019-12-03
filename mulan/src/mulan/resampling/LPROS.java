package mulan.resampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;

/**
<!-- globalinfo-start -->
* Class implementing the Label PowerSet Random Oversampling. For more information, see<br>
* <br>
* Francisco Charte and Antonio Rivera and Maria Jose del Jesus and Francisco Herrera: A first approach to deal with imbalance in multi-label datasets. In: Proc. International Conference on Hybrid Artificial Intelligence Systems, 2013.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* @inproceedings{charte2013first,
*  title={A first approach to deal with imbalance in multi-label datasets},
*  author={Charte, Francisco and Rivera, Antonio and del Jesus, Maria Jose and Herrera, Francisco},
*  booktitle={International Conference on Hybrid Artificial Intelligence Systems},
*  pages={150--160},
*  year={2013},
*  location={La Rioja, Spain}
*}
* </pre>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2017.11.21
*/
public class LPROS extends LPRandomSampling {

	public LPROS(MultiLabelInstances data, String xmlLabels, Double percentage) 
			throws InvalidDataFormatException {
		super(data, xmlLabels, percentage);
	}
	
	public LPROS(MultiLabelInstances data, String xmlLabels) 
			throws InvalidDataFormatException {
		super(data, xmlLabels, 0.25);
	}

	@Override
	public MultiLabelInstances resample() throws InvalidDataFormatException {
		int[] labelIndexes = getData().getLabelIndices();
		Double samplesToCreate = getData().getDataSet().numInstances() * getPercentage();
		HashMap<ArrayList<Integer>, ArrayList<Integer>> labelSetBag = groupSamplesbyLabelSet(labelIndexes);
		Double meanSize = calculateMeanSize(labelSetBag);
		ArrayList<Bag> minBags = new ArrayList<Bag>(); 
		for (ArrayList<Integer> labelSet : labelSetBag.keySet()) {
			ArrayList<Integer> instanceSet = labelSetBag.get(labelSet);
			if (instanceSet.size() < meanSize) {
				Bag newBag = new Bag(instanceSet, BagType.Minority);
				minBags.add(newBag);
			}
		}
		Double meanIncrement = samplesToCreate / minBags.size();
		Collections.sort(minBags);
		Random generator = new Random(System.currentTimeMillis());
		for (int i = 0; i < minBags.size(); i++) {
			Bag minBag = minBags.get(i);
			Double incrementBag = Math.min(meanSize - minBag.getInstances().size(), meanIncrement);
			Double j = 0.0;
			while (j < incrementBag && minBag.getInstances().size() < meanSize) {
				int x = generator.nextInt(minBag.getInstances().size());
				Integer index = minBag.getInstances().get(x);
				Instance instance = getData().getDataSet().get(index);
				getData().getDataSet().add(instance);
				Integer newIndex = getData().getDataSet().size() - 1;
				minBag.addInstance(newIndex);
				minBags.set(i, minBag);
				j++;
			}
			if (j < incrementBag) {
				Double remainder = meanSize - minBag.getInstances().size();
				minBags = distributeAmongBags(minBags, i, remainder, generator);
			}
		}
		return getData();
	}

	@Override
	protected ArrayList<Bag> distributeAmongBags(
			ArrayList<Bag> minBags, int i, 
			Double remainder, Random generator) {
		int j = 0;
		while (remainder > 0) {
			if (j == i) {
				j++;
				continue;
			}
			Bag minBag = minBags.get(j);
			int x = generator.nextInt(minBag.getInstances().size());
			Integer index = minBag.getInstances().get(x);
			Instance instance = getData().getDataSet().get(index);
			getData().getDataSet().add(instance);
			Integer newIndex = getData().getDataSet().size() - 1;
			minBag.addInstance(newIndex);
			minBags.set(j, minBag);
			j++;
			if (j >= minBags.size()){
				j = 0;
			}
			remainder--;
		}
		return minBags;
	}
}
