package mulan.resampling;

import java.util.ArrayList;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;

public abstract class MLRandomSampling {

	private MultiLabelInstances data;
	private String xmlLabels;
	private Double percentage;
	private ImbalanceMetrics metrics;
	
	public MLRandomSampling(MultiLabelInstances data, String xmlLabels, Double percentage)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.percentage = percentage;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public String getXmlLabels() {
		return xmlLabels;
	}

	public void setXmlLabels(String xmlLabels) {
		this.xmlLabels = xmlLabels;
	}
	
	public Double getPercentage() {
		return percentage;
	}

	public void setPercentage(Double percentage) {
		this.percentage = percentage;
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
	
	public ArrayList<Instance> getInstancesFromLabel(Attribute label) {
		ArrayList<Instance> instances = new ArrayList<Instance>(); 
		for (int i = 0; i < getData().getNumInstances(); i++) {
			Instance instance = getData().getDataSet().get(i);
			if (instance.value(label) == 1.0) {
				instances.add(instance);
			}
		}
		return instances;
	}
	
	public abstract MultiLabelInstances resample() throws Exception;
}
