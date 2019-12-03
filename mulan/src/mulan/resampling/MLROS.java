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

/**
<!-- globalinfo-start -->
* Class implementing the MultiLabel Random Oversampling. For more information, see<br>
* <br>
* Francisco Charte and Antonio Rivera and Maria Jose del Jesus and Francisco Herrera: Addressing imbalance in multilabel classification: Measures and random resampling algorithms. In: Neurocomputing, 2015.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* @article{charte2015addressing,
*  title={Addressing imbalance in multilabel classification: Measures and random resampling algorithms},
*  author={Charte, Francisco and Rivera, Antonio and del Jesus, Maria Jose and Herrera, Francisco},
*  journal={Neurocomputing},
*  volume={163},
*  pages={3--16},
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
public class MLROS extends MLRandomSampling {

	public MLROS(MultiLabelInstances data, String xmlLabels, Double percentage)
			throws InvalidDataFormatException {
		super(data, xmlLabels, percentage);
	}
	
	public MLROS(MultiLabelInstances data, String xmlLabels)
			throws InvalidDataFormatException {
		super(data, xmlLabels, 0.25);
	}
	
	@Override
	public MultiLabelInstances resample() throws Exception {
		Double samplesToClone = getData().getNumInstances() * getPercentage();
		Set<Attribute> labelAttributes = getData().getLabelAttributes();
		getMetrics().calculateIRLbl();
		getMetrics().calculateMeanIR();
		HashMap<Attribute, ArrayList<Instance>> minBags = new HashMap<Attribute, ArrayList<Instance>>(); 
		for (Attribute label : labelAttributes) {
			Double iRLBl = getMetrics().getIRLbl().get(label);
			Double meanIR = getMetrics().getMeanIR();
			if (iRLBl != null && iRLBl > meanIR) { 
				minBags.put(label, getInstancesFromLabel(label));
			}
		}
		Random generator = new Random(System.currentTimeMillis());
		while (samplesToClone > 0) {
			ArrayList<Attribute> markedToRemove = new ArrayList<Attribute>();
			for (Attribute label : minBags.keySet()) {
				ArrayList<Instance> minBag = minBags.get(label);
				int sampleIndex = generator.nextInt(minBag.size());
				Instance sampleToDuplicate = minBag.get(sampleIndex);
				Instance newSample = new DenseInstance(sampleToDuplicate);
				getData().getDataSet().add(newSample);
				getMetrics().recalcNewSample(newSample);
				Double newIRLBl = getMetrics().getIRLbl().get(label);
				Double newMeanIR = getMetrics().getMeanIR();
				if (newIRLBl <= newMeanIR) {
					markedToRemove.add(label);
				}
				samplesToClone--;
				if (samplesToClone <= 0) {
					break;
				}
			}
			for (Attribute label : markedToRemove) {
				minBags.remove(label);
			}
			if (minBags.size() == 0) {
				samplesToClone = 0.0;
			}
		}
		return getData(); 
	}
}
