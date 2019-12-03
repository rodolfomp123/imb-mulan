package mulan.resampling;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

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
* @inproceedings{charte2019resampling,
*  title={Tackling multilabel imbalance through label decoupling and data resampling hybridization},
*  author={Charte, Francisco and Rivera, Antonio and del Jesus, Maria Jose and Herrera, Francisco},
*  booktitle={Neurocomputing},
*  pages={110-122},
*  volume={326-327},
*  year={2019}
* }
*
* </pre>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2019.02.13
*/
public class REMEDIALHwR {
	
	private MultiLabelInstances data;
	private String xmlLabels;
	private ImbalanceMetrics metrics;
	private HybridMethod method;

	public REMEDIALHwR(MultiLabelInstances data, String xmlLabels, HybridMethod method)
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
		this.method = method;
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
	
	public MultiLabelInstances resample() throws Exception {
		REMEDIAL remedial = new REMEDIAL(data, xmlLabels);
		MultiLabelInstances remedial_dataset = remedial.resample();
		MultiLabelInstances resampled_dataset = null;
		if (method.equals(HybridMethod.MLeNN)) {
			MLeNN mlenn = new MLeNN(remedial_dataset, xmlLabels);
			resampled_dataset = mlenn.resample();
		} else if (method.equals(HybridMethod.MLSMOTE)) {
			MLSMOTE mlsmote = new MLSMOTE(remedial_dataset, xmlLabels);
			resampled_dataset = mlsmote.resample();
		} else if (method.equals(HybridMethod.MLROS)) {
			MLROS mlros = new MLROS(remedial_dataset, xmlLabels);
			resampled_dataset = mlros.resample();
		}
		return resampled_dataset;
	}

}
