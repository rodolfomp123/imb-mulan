package mulan.resampling;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;

public abstract class LPRandomSampling {

	private MultiLabelInstances data;
	private Double percentage;
	private String xmlLabels;
	
	public LPRandomSampling(MultiLabelInstances data, String xmlLabels, Double percentage)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.percentage = percentage;
	}

	public MultiLabelInstances getData() {
		return data;
	}

	public void setData(MultiLabelInstances data) {
		this.data = data;
	}
	
	public Double getPercentage() {
		return percentage;
	}

	public void setPercentage(Double percentage) {
		this.percentage = percentage;
	}

	public String getXmlLabels() {
		return xmlLabels;
	}

	public void setXmlLabels(String xmlLabels) {
		this.xmlLabels = xmlLabels;
	}
	
	public abstract MultiLabelInstances resample() throws InvalidDataFormatException;

	protected abstract ArrayList<Bag> distributeAmongBags(
			ArrayList<Bag> bags, int i, 
			Double remainder, Random generator);

	protected Double calculateMeanSize(HashMap<ArrayList<Integer>, ArrayList<Integer>> labelSetBag) {
		Collection<ArrayList<Integer>> instanceSets = labelSetBag.values();
		Double total = 0.0;
		for (ArrayList<Integer> instanceSet : instanceSets) {
			total += instanceSet.size();
		}
		return total / instanceSets.size();
	}
	
	protected HashMap<ArrayList<Integer>, ArrayList<Integer>> groupSamplesbyLabelSet(int[] labelIndexes) {
		HashMap<ArrayList<Integer>, ArrayList<Integer>> labelSetBag = 
				new HashMap<ArrayList<Integer>, ArrayList<Integer>>();
		for (int i = 0; i < data.getDataSet().numInstances(); i++) {
			Instance instance = data.getDataSet().get(i);
			ArrayList<Integer> labelBag = getLabels(instance, labelIndexes);
			ArrayList<Integer> instances = labelSetBag.get(labelBag);
			if (instances != null) {
				instances.add(i);
				labelSetBag.replace(labelBag, instances);
			} else {
				ArrayList<Integer> newList = new ArrayList<Integer>();
				newList.add(i);
				labelSetBag.put(labelBag, newList);	
			}
		}
		return labelSetBag;
	}

	protected ArrayList<Integer> getLabels(Instance instance, int[] labelIndexes) {
		ArrayList<Integer> labelBag = new ArrayList<Integer>();  
		for (int index : labelIndexes) {
			if (instance.value(index) == 1.0) {
				labelBag.add(index);
			}
		}
		return labelBag;
	}
}
