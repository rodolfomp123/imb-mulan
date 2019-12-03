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
* @version 2017.11.23
*/
public class LPRUS extends LPRandomSampling {
	
	private ArrayList<Instance> instancesToRemove = new ArrayList<Instance>();

	public void addInstancesToRemove(Instance instance) {
		this.instancesToRemove.add(instance);
	}
	
	public ArrayList<Instance> getInstancesToRemove() {
		return this.instancesToRemove;
	}

	public LPRUS(MultiLabelInstances data, String xmlLabels, Double percentage) 
			throws InvalidDataFormatException {
		super(data, xmlLabels, percentage);
	}
	
	public LPRUS(MultiLabelInstances data, String xmlLabels) 
			throws InvalidDataFormatException {
		super(data, xmlLabels, 0.25);
	}

	@Override
	public MultiLabelInstances resample() throws InvalidDataFormatException {
		int[] labelIndexes = getData().getLabelIndices();
		Double samplesToDelete = getData().getDataSet().numInstances() * getPercentage();
		HashMap<ArrayList<Integer>, ArrayList<Integer>> labelSetBag = groupSamplesbyLabelSet(labelIndexes);
		Double meanSize = calculateMeanSize(labelSetBag);
		ArrayList<Bag> majBags = new ArrayList<Bag>(); 
		for (ArrayList<Integer> labelSet : labelSetBag.keySet()) {
			ArrayList<Integer> instanceSet = labelSetBag.get(labelSet);
			if (instanceSet.size() > meanSize) {
				Bag newBag = new Bag(instanceSet, BagType.Majority);
				majBags.add(newBag);
			}
		}
		Double meanReduction = samplesToDelete / majBags.size();
		Collections.sort(majBags);
		Random generator = new Random(System.currentTimeMillis());
		for (int i = 0; i < majBags.size(); i++) {
			Bag majBag = majBags.get(i);
			Double reductionBag = Math.min(majBag.getInstances().size() - meanSize, meanReduction);
			for (Double j = 0.0; j < reductionBag; j++) {
				int x = generator.nextInt(majBag.getInstances().size());
				Integer index = majBag.getInstances().get(x);
				addInstancesToRemove(getData().getDataSet().get(index));
				majBag.removeInstance(index);
				majBags.set(i, majBag);
			}
			if (majBag.getInstances().size() >= meanSize) {
				Double remainder = majBag.getInstances().size() - reductionBag;
				majBags = distributeAmongBags(majBags, i, remainder, generator);
			}
		}
		for (Instance instance : getInstancesToRemove()) {
			getData().getDataSet().remove(instance);
		}
		return getData();
	}

	@Override
	protected ArrayList<Bag> distributeAmongBags(
			ArrayList<Bag> majBags, int i, 
			Double remainder, Random generator) {
		int j = i + 1;
		while (remainder > 0 && j < majBags.size()) {
			Bag majBag = majBags.get(j);
			int x = generator.nextInt(majBag.getInstances().size());
			Integer index = majBag.getInstances().get(x);
			getInstancesToRemove().add(getData().getDataSet().get(index));
			majBag.removeInstance(index);
			majBags.set(j, majBag);
			j++;
			remainder--;
		}
		return majBags;
	}
}
