package mulan.resampling;

import java.util.HashMap;
import java.util.Set;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class ImbalanceMetrics {

	private Instances dataset;
	private Set<Attribute> labels;
	
	private HashMap<Attribute, Double> hSumPerLbl;
	private Double maxHSum;
	
	private Double meanIR;
	private HashMap<Attribute, Double> IRLbl;
	
	private HashMap<Instance, Double> instScumble;
	private Double scumble;
	
	public ImbalanceMetrics(Instances dataset, Set<Attribute> labels) {
		this.dataset = dataset;
		this.labels = labels;
	}
	
	public ImbalanceMetrics(ImbalanceMetrics metrics) {
		this.IRLbl = metrics.getIRLbl();
		this.meanIR = metrics.getMeanIR();
		this.hSumPerLbl = metrics.gethSumPerLbl();
		this.maxHSum = metrics.getMaxHSum();
	}

	public Instances getDataset() {
		return dataset;
	}

	public void setDataset(Instances dataset) {
		this.dataset = dataset;
	}

	public Set<Attribute> getLabels() {
		return labels;
	}

	public void setLabels(Set<Attribute> labels) {
		this.labels = labels;
	}
	
	public Double getMeanIR() {
		return meanIR;
	}
	
	public void setMeanIR(Double meanIR) {
		this.meanIR = meanIR;
	}
	
	public HashMap<Attribute, Double> getIRLbl() {
		return IRLbl;
	}
	
	public void setIRLbl(HashMap<Attribute, Double> iRLbl) {
		IRLbl = iRLbl;
	}

	public HashMap<Attribute, Double> gethSumPerLbl() {
		return hSumPerLbl;
	}

	public void sethSumPerLbl(HashMap<Attribute, Double> hSumPerLbl) {
		this.hSumPerLbl = hSumPerLbl;
	}

	public Double getMaxHSum() {
		return maxHSum;
	}

	public void setMaxHSum(Double maxHSum) {
		this.maxHSum = maxHSum;
	}
	
	public HashMap<Instance, Double> getInstScumble() {
		return instScumble;
	}

	public void setInstScumble(HashMap<Instance, Double> instScumble) {
		this.instScumble = instScumble;
	}

	public Double getScumble() {
		return scumble;
	}

	public void setScumble(Double scumble) {
		this.scumble = scumble;
	}
	
	private boolean hasAttribute(Instance instance, Attribute attr) {
		for (int i = 0; i < instance.numAttributes(); i++) {
			Attribute attribute = instance.attribute(i);
			if (attribute.equals(attr) && instance.value(attribute) == 1.0) {
				return true;
			}
		}
		return false;
	}
	
	public Double calculateScumble() {
		Double sumScumble = 0.0, scumble_i = 0.0;
		instScumble = new HashMap<Instance, Double>();
		for (Instance instance : dataset) {
			Double sumIRLbl = 0.0, numLabInst = 0.0, prod = 1.0;
			for (Attribute attribute : labels) {
				Double iRLbl = 1.0;
				if (instance.value(attribute) == 1.0) {
					iRLbl = IRLbl.get(attribute);
					sumIRLbl += iRLbl;
					numLabInst++;
				}
				prod *= iRLbl;
			}
			Double iRLbl_i = sumIRLbl/numLabInst;
			scumble_i = 1.0 - ((1.0/iRLbl_i) * Math.pow(prod, 1.0/numLabInst));
			instScumble.put(instance, scumble_i);
			sumScumble += scumble_i;
		}
		Double value = sumScumble/dataset.size();
		this.scumble = value;
		return value;
	}
	
	public Double calculateMeanIR() {
        Double sum = 0.0;
        Double countZeros = 0.0;
        for (Attribute label : labels) {
        	Double ir = IRLbl.get(label);
        	if (ir != null) {
        		sum += IRLbl.get(label);
        	} else {
        		countZeros++;
        	}
        }
        Double numberOfLabels = new Double(labels.size() - countZeros);
        meanIR = sum / numberOfLabels;
        return meanIR;
    }
	
    public void calculateIRLbl() {
        maxHSum = -1.0;
        hSumPerLbl = new HashMap<Attribute, Double>();
        for (Attribute label : labels) {
            Double hSum = 0.0;
            for (Instance instance : dataset) {
                if (hasAttribute(instance, label)) {
                    hSum++;
                }
            }
            if (hSum != 0.0) {
            	hSumPerLbl.put(label, hSum);
            	if (hSum > maxHSum) {
                    maxHSum = hSum;
                }
            }
        }
        IRLbl = new HashMap<Attribute, Double>();
        for (Attribute label : labels) {
            Double hSum = hSumPerLbl.get(label);
            if (hSum != null) {
            	IRLbl.put(label, maxHSum / hSum);
            }
        }
    }
	
    public void recalcNewSample(Instance newSample) {
    	for (Attribute label : labels) {
    		if (newSample.value(label) == 1.0) {
    			Double newHSum = hSumPerLbl.get(label) + 1;
    			hSumPerLbl.replace(label, newHSum);
    			if (newHSum > maxHSum) {
    				maxHSum = newHSum;
    			}
    			Double newIRLbl = maxHSum / newHSum;
    			IRLbl.replace(label, newIRLbl);
    		}
    	}
    	calculateMeanIR();
    }
    
	public void recalcDelSample(Instance delSample) {
		for (Attribute label : labels) {
    		if (delSample.value(label) == 1.0) {
    			Double newHSum = hSumPerLbl.get(label) - 1;
    			hSumPerLbl.replace(label, newHSum);
    			if (maxHSum == newHSum + 1) {
    				lookForNewMax();
    			}
    			Double newIRLbl = maxHSum / newHSum;
    			IRLbl.replace(label, newIRLbl);
    		}
    	}
    	calculateMeanIR();
	}

    
    private void lookForNewMax() {
    	for (Attribute label : hSumPerLbl.keySet()) {
    		Double hSum = hSumPerLbl.get(label);
    		if (hSum > maxHSum) {
    			maxHSum = hSum;
    		}
    	}
	}

	public Double getMaxIR() {
    	Double maxIR = 0.0;
    	for (Attribute label : IRLbl.keySet()) {
    		Double currentIR = IRLbl.get(label);
    		if (currentIR > maxIR) {
    			maxIR = currentIR;
    		}
    	}
    	return maxIR;
    }
}
