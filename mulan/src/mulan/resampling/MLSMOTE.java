package mulan.resampling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
<!-- globalinfo-start -->
* Class implementing the Multilabel Synthetic Minority Oversampling Technique. For more information, see<br>
* <br>
* Francisco Charte and Antonio Rivera and Maria Jose del Jesus and Francisco Herrera: MLSMOTE: approaching imbalanced multilabel learning through synthetic instance generation. In: Knowledge-Based Systems, 2015.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* @article{charte2015mlsmote,
*  title={MLSMOTE: approaching imbalanced multilabel learning through synthetic instance generation},
*  author={Charte, Francisco and Rivera, Antonio and del Jesus, Maria Jose and Herrera, Francisco},
*  journal={Knowledge-Based Systems},
*  volume={89},
*  pages={385--397},
*  year={2015},
*  publisher={Elsevier}
*}
*
* </pre>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2017.11.24
*/
public class MLSMOTE {

	private MultiLabelInstances data;
	private Integer nn;
	private String xmlLabels;
	private Double percentage;
	private LabelCombination labelCombination;
	private ImbalanceMetrics metrics;
	
	public MLSMOTE(MultiLabelInstances data, String xmlLabels, Integer nn, 
			LabelCombination labelCombination, Double percentage) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = nn;
		this.percentage = percentage;
		this.labelCombination = labelCombination;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLSMOTE(MultiLabelInstances data, String xmlLabels, 
			LabelCombination labelCombination, Double percentage) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.labelCombination = labelCombination;
		this.percentage = percentage;
		this.nn = 5;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLSMOTE(MultiLabelInstances data, String xmlLabels, Integer nn, Double percentage)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = nn;
		this.percentage = percentage;
		this.labelCombination = LabelCombination.Ranking;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLSMOTE(MultiLabelInstances data, String xmlLabels, Double percentage)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.percentage = percentage;
		this.nn = 5;
		this.labelCombination = LabelCombination.Ranking;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLSMOTE(MultiLabelInstances data, String xmlLabels)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.percentage = 0.25;
		this.nn = 5;
		this.labelCombination = LabelCombination.Ranking;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}

	public Double getPercentage() {
		return percentage;
	}

	public void setPercentage(Double percentage) {
		this.percentage = percentage;
	}
	
	public ImbalanceMetrics getMetrics() {
		return metrics;
	}

	public void setMetrics(ImbalanceMetrics metrics) {
		this.metrics = metrics;
	}
	
	public MultiLabelInstances getData() {
		return data;
	}

	public void setData(MultiLabelInstances data) {
		this.data = data;
	}

	public String getXmlLabels() {
		return xmlLabels;
	}

	public void setXmlLabels(String xmlLabels) {
		this.xmlLabels = xmlLabels;
	}

	public LabelCombination getLabelCombination() {
		return labelCombination;
	}

	public void setLabelCombination(LabelCombination labelCombination) {
		this.labelCombination = labelCombination;
	}
	
	public Integer getNn() {
		return nn;
	}

	public void setNn(Integer nn) {
		this.nn = nn;
	}
	
	private Instances nearestNeighbors(Instance instance, Instances dataSet) throws Exception {
		LinearNNSearch knn = new LinearNNSearch(dataSet);
		Instances nearestInstances = knn.kNearestNeighbours(instance, getNn());		
		return nearestInstances;
	}
	
	public MultiLabelInstances resample() throws Exception {
		Set<Attribute> labelSet = getData().getLabelAttributes();
		getMetrics().calculateIRLbl();
		getMetrics().calculateMeanIR();
		Random generator = new Random(System.currentTimeMillis());
		ArrayList<Instance> newInstances = new ArrayList<Instance>(); 
		for (Attribute label : labelSet) {
			Double iRLbl = getMetrics().getIRLbl().get(label);
			Double meanIR = getMetrics().getMeanIR();
			if (iRLbl != null && iRLbl > meanIR) {
				ArrayList<Integer> minBag = getAllInstancesOfLabel(label);
				for (Integer index : minBag) {
					Instance sample = getData().getDataSet().get(index);
					Instances neighbors = nearestNeighbors(sample, getData().getDataSet());
					int randIndex = generator.nextInt(neighbors.size());
					Instance refNeigh = neighbors.get(randIndex);
					Instance synthSample = newSample(sample, refNeigh, neighbors, generator);
					newInstances.add(synthSample);
				}
			}
		}
		Double samplesToCreate = getPercentage() * getData().getDataSet().numInstances();
		int i = 0, n = 0;
		while (n < samplesToCreate && i < newInstances.size()) {
			getData().getDataSet().add(newInstances.get(i));
			i++;
			n++;
		}
		return getData();
	}

	private Instance newSample(Instance sample, Instance refNeigh, Instances neighbors, Random generator) {
		Instance synthSample = new DenseInstance(sample);
		for (Attribute feature : getData().getFeatureAttributes()) {
			if (feature.isNumeric()) {
				Double diff = refNeigh.value(feature) - sample.value(feature);
				Double offset = diff * generator.nextInt(1);
				Double value = sample.value(feature) + offset;
				synthSample.setValue(feature, value);
			} else {
				String value = mostFreqVal(neighbors, feature);
				synthSample.setValue(feature, value);
			}
		}
		ArrayList<Attribute> newLabelSet = createLabelSet(sample, neighbors);
		for (Attribute label : getData().getLabelAttributes()) {
			if (newLabelSet.contains(label)) {
				synthSample.setValue(label, 1);
			} else {
				synthSample.setValue(label, 0);
			}
		}
		return synthSample;
	}

	private ArrayList<Attribute> createLabelSet(Instance sample, Instances neighbors) {
		ArrayList<Instance> instances = new ArrayList<Instance>();
		instances.addAll(neighbors);
		instances.add(sample);
		if (getLabelCombination() == LabelCombination.Ranking) {
			return createLabelRanking(instances);
		} else if (getLabelCombination() == LabelCombination.Intersection) {
			return createLabelIntersection(instances);
		} else {
			return createLabelUnion(instances);
		}
	}
	
	private ArrayList<Attribute> createLabelIntersection(ArrayList<Instance> neighbors) {
		int numInstances = neighbors.size();
		int numLabels = getData().getNumLabels();
		int[] labelIndices = getData().getLabelIndices();
		int[] intersection = null;
		for (int i = 0; i < numInstances; i++) {
			int[] labels = new int[numLabels];
			for (int j = 0; j < numLabels; j++) {
				if (neighbors.get(i).stringValue(labelIndices[j]).equals("1")) {
					labels[j] = 1;
				} else {
					labels[j] = 0;
				}
			}
			if (intersection == null) {
				intersection = labels;
			} else {
				for (int k = 0; k < numLabels; k++) {
					intersection[k] = intersection[k] & labels[k];
				}
			}
		}
		ArrayList<Attribute> labels = new ArrayList<Attribute>();
		ArrayList<Attribute> labelAttributes = new ArrayList<Attribute>();
		labelAttributes.addAll(getData().getFeatureAttributes());
		for (int k = 0; k < numLabels; k++) {
			if (intersection[k] == 1) {
				labels.add(labelAttributes.get(k));
			}
		}
		return labels;
	}

	private ArrayList<Attribute> createLabelUnion(ArrayList<Instance> neighbors) {
		int numInstances = neighbors.size();
		int numLabels = getData().getNumLabels();
		int[] labelIndices = getData().getLabelIndices();
		int[] union = new int[numLabels];
		int[] dblLabels = new int[numLabels];
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (neighbors.get(i).stringValue(labelIndices[j]).equals("1")) {
					dblLabels[j] = 1;
				} else {
					dblLabels[j] = 0;
				}
			}
			for (int index = 0; index < union.length; index++) {
				union[index] = ((union[index] == 1) || (dblLabels[index] == 1) ? 1 : 0);
			}
		}
		ArrayList<Attribute> labels = new ArrayList<Attribute>();
		ArrayList<Attribute> labelAttributes = new ArrayList<Attribute>();
		labelAttributes.addAll(getData().getFeatureAttributes());
		for (int k = 0; k < numLabels; k++) {
			if (union[k] == 1) {
				labels.add(labelAttributes.get(k));
			}
		}
		return labels;
	}
	
	private ArrayList<Attribute> createLabelRanking(ArrayList<Instance> neighbors) {
		int numInstances = neighbors.size();
		int numLabels = getData().getNumLabels();
		int[] labelIndices = getData().getLabelIndices();
		int[] labelCount = new int[numLabels];
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (neighbors.get(i).stringValue(labelIndices[j]).equals("1")) {
					labelCount[j] += 1;
				}
			}
		}
		ArrayList<Attribute> labels = new ArrayList<Attribute>();
		ArrayList<Attribute> labelAttributes = new ArrayList<Attribute>();
		labelAttributes.addAll(getData().getLabelAttributes());
		int threshold = (int)(neighbors.size() * 0.5);
		for (int j = 0; j < numLabels; j++) {
			if (labelCount[j] >= threshold) {
				labels.add(labelAttributes.get(j));
			}
		}
		return labels;
	}

	private String mostFreqVal(Instances neighbors, Attribute feature) {
		HashMap<String, Integer> countMap = new HashMap<String, Integer>(); 
		for (Instance sample : neighbors) {
			String value = sample.stringValue(feature);
			Integer count = countMap.get(value);
			if (count == null) {
				countMap.put(value, 1);
			} else {
				count++;
				countMap.replace(value, count);
			}
		}
		Integer higherCount = 0;
		String mostFreq = null;
		for (String key : countMap.keySet()) {
			Integer count = countMap.get(key);
			if (count > higherCount) {
				higherCount = count;
				mostFreq = key;
			}
		}
		return mostFreq;
	}

	private ArrayList<Integer> getAllInstancesOfLabel(Attribute label) {
		ArrayList<Integer> instanceIndices = new ArrayList<Integer>(); 
		for (int i = 0; i < getData().getDataSet().size(); i++) {
			Instance sample = getData().getDataSet().get(i);
			if (sample.value(label) == 1.0) {
				instanceIndices.add(i);
			}
		}
		return instanceIndices;
	}
}
