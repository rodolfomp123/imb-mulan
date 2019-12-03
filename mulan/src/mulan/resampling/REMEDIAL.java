package mulan.resampling;

import java.util.ArrayList;
import java.util.Set;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
<!-- globalinfo-start -->
* Class implementing the MultiLabel Resampling Method based on the oncurrence among Imbalanced Labels. For more information, see<br>
* <br>
* Francisco Charte and Antonio Rivera and Maria Jose del Jesus and Francisco Herrera: Resampling multilabel datasets by decoupling highly imbalanced labels. In: International Conference on Hybrid Artificial Intelligence Systems, 2015.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* @inproceedings{charte2015resampling,
*  title={Resampling multilabel datasets by decoupling highly imbalanced labels},
*  author={Charte, Francisco and Rivera, Antonio and del Jesus, Maria Jose and Herrera, Francisco},
*  booktitle={International Conference on Hybrid Artificial Intelligence Systems},
*  pages={489--501},
*  year={2015},
*  organization={Springer}
* }
*
* </pre>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2017.11.29
*/
public class REMEDIAL {
	
	private MultiLabelInstances data;
	private String xmlLabels;
	private ImbalanceMetrics metrics;

	public REMEDIAL(MultiLabelInstances data, String xmlLabels)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
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
	
	public ImbalanceMetrics getMetrics() {
		return metrics;
	}

	public void setMetrics(ImbalanceMetrics metrics) {
		this.metrics = metrics;
	}
	
	public MultiLabelInstances resample() throws InvalidDataFormatException {
		Set<Attribute> labels = getData().getLabelAttributes();
		getMetrics().calculateIRLbl();
		getMetrics().calculateMeanIR();
		getMetrics().calculateScumble();
		ArrayList<Instance> newInstances = new ArrayList<Instance>();
		for (int i = 0; i < getData().getDataSet().size(); i++) {
			Instance instance = getData().getDataSet().get(i);
			Double scumble_i = getMetrics().getInstScumble().get(instance);
			if (scumble_i > getMetrics().getScumble()) {
				Instance newInstance = new DenseInstance(instance);
				for (Attribute label : labels) {
					if (instance.value(label) == 1.0) {
						Double irlbl = getMetrics().getIRLbl().get(label);
						if (irlbl != null) { 
							if (irlbl <= getMetrics().getMeanIR()) {
								instance.setValue(label, 0.0);
								getData().getDataSet().set(i, instance);
							} else {
								newInstance.setValue(label, 0.0);
							}
						}
					}
				}
				newInstances.add(newInstance);
			}
		}
		for (Instance newInstance : newInstances) {
			getData().getDataSet().add(newInstance);
		}
		return getData();
	}

}
