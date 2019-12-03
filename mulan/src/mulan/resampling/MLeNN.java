package mulan.resampling;

import java.util.ArrayList;
import java.util.Set;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
<!-- globalinfo-start -->
* Class implementing the MultiLabel edited Nearest Neighbor Undersampling. For more information, see<br>
* <br>
* Francisco Charte and Antonio Rivera and Maria Jose del Jesus and Francisco Herrera: MLeNN: a first approach to heuristic multilabel undersampling. In: Proc. International Conference on Intelligent Data Engineering and Automated Learning, 2014.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* @inproceedings{charte2014mlenn,
*  title={MLeNN: a first approach to heuristic multilabel undersampling},
*  author={Charte, Francisco and Rivera, Antonio and del Jesus, Maria Jose and Herrera, Francisco},
*  booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
*  pages={1--9},
*  year={2014},
*  location={Wroclaw, Poland}
*}
*
* </pre>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2017.11.15
*/
public class MLeNN {

	private MultiLabelInstances data;
	private String xmlLabels;
	private Double ht;
	private Integer nn;
	private ImbalanceMetrics metrics;
	
	public MLeNN(MultiLabelInstances data, String xmlLabels, Double ht, Integer nn) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.ht = ht;
		this.nn = nn;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLeNN(MultiLabelInstances data, String xmlLabels, Double ht) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.ht = ht;
		this.nn = 3;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLeNN(MultiLabelInstances data, String xmlLabels, Integer nn) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = nn;
		this.ht = 0.5;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLeNN(MultiLabelInstances data, String xmlLabels) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = 3;
		this.ht = 0.5;
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
	
	public Double getHt() {
		return ht;
	}

	public void setHt(Double ht) {
		this.ht = ht;
	}

	public Integer getNn() {
		return nn;
	}

	public void setNn(Integer nn) {
		this.nn = nn;
	}
	
	public String getXmlLabels() {
		return xmlLabels;
	}

	public void setXmlLabels(String xmlLabels) {
		this.xmlLabels = xmlLabels;
	}
	
	private Instances nearestNeighbors(Instance instance, Instances dataSet) throws Exception {
		LinearNNSearch knn = new LinearNNSearch(dataSet);
		Instances nearestInstances = knn.kNearestNeighbours(instance, getNn());		
		return nearestInstances;
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
	
	public MultiLabelInstances resample() throws Exception {
		Set<Attribute> labelAttributes = getData().getLabelAttributes();
		getMetrics().calculateIRLbl();
		getMetrics().calculateMeanIR();
		ArrayList<Instance> instancesToRemove = new ArrayList<Instance>(); 
		for (int index = 0; index < getData().getDataSet().numInstances(); index++) {
			Instance instance = getData().getDataSet().get(index);
			boolean nextSample = false;
			for (Attribute label : labelAttributes) {
				double value = instance.value(label);
				if (value == 1.0) {
					Double iRLbl = getMetrics().getIRLbl().get(label);
					if (iRLbl != null && iRLbl > getMetrics().getMeanIR()) {
						nextSample = true;
					}
				}
			}
			if (nextSample) {
				continue;
			}
			Integer numDifferences = 0;
			for (Instance neighbor : nearestNeighbors(instance, getData().getDataSet())) {
				if (adjustedHammingDist(instance, neighbor, getData().getLabelAttributes()) > getHt()) {
					numDifferences++;
				}
			}
			if (numDifferences >= getNn()/2) {
				instancesToRemove.add(instance);
			}
		}
		for (Instance sample : instancesToRemove) {
			getData().getDataSet().remove(sample);
		}
		return getData(); 
	}
}
