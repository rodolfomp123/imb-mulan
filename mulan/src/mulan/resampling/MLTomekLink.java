package mulan.resampling;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Set;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;

/**
<!-- globalinfo-start -->
* Class implementing the MultiLabel Tomek Link Undersampling. For more information, see<br>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2017.11.15
*/
public class MLTomekLink {

	private MultiLabelInstances data;
	private String xmlLabels;
	private Double threshold;
	private Boolean isCleaningMethod;
	private ImbalanceMetrics metrics;
	
	public MLTomekLink(MultiLabelInstances data, String xmlLabels)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.isCleaningMethod = Boolean.FALSE;
		this.xmlLabels = xmlLabels;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLTomekLink(MultiLabelInstances data, String xmlLabels, Double threshold)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.threshold = threshold;
		this.isCleaningMethod = Boolean.FALSE;
		this.xmlLabels = xmlLabels;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLTomekLink(MultiLabelInstances data, String xmlLabels, Boolean isCleaningMethod)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.isCleaningMethod = isCleaningMethod;
		this.xmlLabels = xmlLabels;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLTomekLink(
			MultiLabelInstances data, String xmlLabels, Double threshold, Boolean isCleaningMethod)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.threshold = threshold;
		this.isCleaningMethod = isCleaningMethod;
		this.xmlLabels = xmlLabels;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MultiLabelInstances getData() {
		return data;
	}

	public void setData(MultiLabelInstances data) {
		this.data = data;
	}

	public ImbalanceMetrics getMetrics() {
		return metrics;
	}

	public void setMetrics(ImbalanceMetrics metrics) {
		this.metrics = metrics;
	}
	
	public Boolean isCleaningMethod() {
		return isCleaningMethod;
	}

	public void setCleaningMethod(Boolean isCleaningMethod) {
		this.isCleaningMethod = isCleaningMethod;
	}
	
	public Double getThreashold() {
		return threshold;
	}

	public void setThreashold(Double threshold) {
		this.threshold = threshold;
	}
	
	public String getXmlLabels() {
		return xmlLabels;
	}

	public void setXmlLabels(String xmlLabels) {
		this.xmlLabels = xmlLabels;
	}
	
	private Double calculateThreshold() {
		Double indicator = 1.0 / Math.sqrt(getMetrics().getMeanIR());
		Double th = 0.0;
		if (indicator > 0.5) {
			th = 0.5;
		} else if (indicator > 0.3) {
			th = 0.3;
		} else {
			th = 0.15;
		}
		Double scumble = metrics.getScumble();
		if (scumble > 0.3) {
			return 0.5;
		}
		return th;
	}
	
	private Double adjustedHammingDist(Instance instance, Instance neighbor, Set<Attribute> labels) {
		Double distance = 0.0;
		Double num_labels = 0.0;
		for (Attribute label : labels) {
			if (instance.value(label) != neighbor.value(label)) {
				distance++;
			}
			if (instance.value(label) == 1.0 || neighbor.value(label) == 1.0) {
				num_labels++;
			}
		}
		return distance / num_labels;
	}
	
	private Instance nearestNeighbor(Instance instance, EuclideanDistance ed) throws Exception {
		Instance nearestInstance = null;
		Double closest = null;
		for (Instance sample : getData().getDataSet()) {
			if (sample.equals(instance)) {
				continue;
			}
			double distance = ed.distance(instance, sample);
			if (closest == null || distance < closest) {
				closest = distance;
				nearestInstance = sample;
			}
		}
		return nearestInstance;
	}
    
	private ArrayList<Instance> getInstancesFromLabel(Attribute label) {
		ArrayList<Instance> instances = new ArrayList<Instance>(); 
		for (int i = 0; i < getData().getNumInstances(); i++) {
			Instance instance = getData().getDataSet().get(i);
			if (instance.value(label) == 1.0) {
				instances.add(instance);
			}
		}
		return instances;
	}
	
	public MultiLabelInstances resample() throws Exception {
		getMetrics().calculateIRLbl();
		getMetrics().calculateMeanIR();
		getMetrics().calculateScumble();
		if (threshold == null) {
			threshold = calculateThreshold();
		}
		ArrayList<Instance> tomekLinks = retrieveTomekLinks();
		for (Instance sample : tomekLinks) {
			getData().getDataSet().remove(sample);
		}
		return getData(); 
	}

	private ArrayList<Instance> retrieveTomekLinks() throws Exception {
		Set<Attribute> labelAttributes = getData().getLabelAttributes();
		EuclideanDistance ed = new EuclideanDistance(getData().getDataSet());
		ArrayList<Instance> tomekLinks = null;
		if (isCleaningMethod()) {
			tomekLinks = retriveAllTomekLinks(labelAttributes, ed);
		} else {
			tomekLinks = retriveMajorityTomekLinks(labelAttributes, ed);
		}
		if (tomekLinks.size() == getData().getNumInstances()) {
			System.out.println("Your dataset has a high SCUMBLE value.");
			System.out.println("You may considers use REMEDIAL algorithm first or use a higher HT value.");
		}
		return tomekLinks;
	}

	private ArrayList<Instance> retriveAllTomekLinks(
			Set<Attribute> labelAttributes, EuclideanDistance ed) throws Exception {
		ArrayList<Instance> tomekLinks = new ArrayList<Instance>();
		ArrayList<Instance> checkedSamples = new ArrayList<Instance>();
		for (Instance sample : getData().getDataSet()) {
			if (checkedSamples.contains(sample)) {
				continue;
			}
			Instance nearestNeighbor = nearestNeighbor(sample, ed);
			checkedSamples.add(sample);
			Double distance = adjustedHammingDist(sample, nearestNeighbor, labelAttributes);
			if (distance > getThreashold()) {
				tomekLinks.add(sample);
			}
		}
		return tomekLinks;
	}

	private ArrayList<Instance> retriveMajorityTomekLinks(
			Set<Attribute> labelAttributes, EuclideanDistance ed) throws Exception {
		ArrayList<Instance> tomekLinks = new ArrayList<Instance>();
		HashMap<Attribute, ArrayList<Instance>> majBags = getMajorityBag(labelAttributes);
		Collection<ArrayList<Instance>> values = majBags.values();
		ArrayList<Instance> checkedSamples = new ArrayList<Instance>();
		for (ArrayList<Instance> majBag : values) {
			for (Instance sample : majBag) {
				if (checkedSamples.contains(sample)) {
					continue;
				}
				Instance nearestNeighbor = nearestNeighbor(sample, ed);
				checkedSamples.add(sample);
				Double distance = adjustedHammingDist(sample, nearestNeighbor, labelAttributes);
				if (distance > getThreashold()) {
					tomekLinks.add(sample);
				}
			}
		}
		return tomekLinks;
	}

	private HashMap<Attribute, ArrayList<Instance>> getMajorityBag(Set<Attribute> labelAttributes) {
		HashMap<Attribute, ArrayList<Instance>> majBags = new HashMap<Attribute, ArrayList<Instance>>(); 
		for (Attribute label : labelAttributes) {
			Double iRLBl = getMetrics().getIRLbl().get(label);
			Double meanIR = getMetrics().getMeanIR();
			if (iRLBl != null && iRLBl < meanIR) { 
				majBags.put(label, getInstancesFromLabel(label));
			}
		}
		return majBags;
	}
}
