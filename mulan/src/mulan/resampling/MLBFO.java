package mulan.resampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
<!-- globalinfo-start -->
* Class implementing the Best-first Oversampling. For more information, see<br>
* <br>
* Xusheng Ai and Jian Wu and Victor Sheng and Yufeng Yao and Pengpeng Zhao and Zhiming Cui: Best first over-sampling for multilabel classification. In: Proc. ACM International on Conference on Information and Knowledge Management, 2015.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* @inproceedings{ai2015best,
*  title={Best first over-sampling for multilabel classification},
*  author={Ai, Xusheng and Wu, Jian and Sheng, Victor S and Yao, Yufeng and Zhao, Pengpeng and Cui, Zhiming},
*  booktitle={ACM International on Conference on Information and Knowledge Management},
*  pages={1803--1806},
*  year={2015},
*  location={Melbourne, Australia}
*}
*
* </pre>
* <br>
<!-- technical-bibtex-end -->
*
* @author Rodolfo Miranda Pereira
* @version 2017.11.23
*/
public class MLBFO {

	private MultiLabelInstances data;
	private String xmlLabels;
	private Double percentage;
	private Integer nn;
	private ImbalanceMetrics metrics;
	
	public MLBFO(MultiLabelInstances data, String xmlLabels, Integer nn, Double percentage) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = nn;
		this.percentage = percentage;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLBFO(MultiLabelInstances data, String xmlLabels, Double percentage) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = 3;
		this.percentage = percentage;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}
	
	public MLBFO(MultiLabelInstances data, String xmlLabels) 
			throws InvalidDataFormatException {
		this.data = new MultiLabelInstances(data.getDataSet(), xmlLabels);
		this.xmlLabels = xmlLabels;
		this.nn = 3;
		this.percentage = 0.25;
		this.metrics = new ImbalanceMetrics(data.getDataSet(), data.getLabelAttributes());
	}

	public ImbalanceMetrics getMetrics() {
		return metrics;
	}

	public void setMetrics(ImbalanceMetrics metrics) {
		this.metrics = metrics;
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

	public String getXmlLabels() {
		return xmlLabels;
	}

	public void setXmlLabels(String xmlLabels) {
		this.xmlLabels = xmlLabels;
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
		ArrayList<Node> tree = contructSortedTree();
		Instances newDataSet = getData().getDataSet();
		getMetrics().calculateIRLbl();
		getMetrics().calculateMeanIR();
		double samplesToDuplicate = getPercentage() * getData().getNumInstances();
		int n = 0;
		while (n < samplesToDuplicate) {
			ArrayList<Instance> candidates = getUnvisitedInstances(tree);
			int i = 0;
			while (i < candidates.size() && n < samplesToDuplicate) { 
				Instance candidate = candidates.get(i);
				i++;
				Instances combinedDataset = new Instances(newDataSet);
				combinedDataset.add(candidate);
				ImbalanceMetrics newImbMetrics = new ImbalanceMetrics(getMetrics());
				newImbMetrics.recalcNewSample(candidate);
				if (newImbMetrics.getMeanIR() < metrics.getMeanIR()) {
					continue;
				}
				Instances nearNeigh = nearestNeighbors(candidate, combinedDataset);
				if (numberNNSharingLabel(nearNeigh) < nearNeigh.size() / 2) {
					continue;
				}
				if (newImbMetrics.getMeanIR() > metrics.getMeanIR()) {
					metrics = newImbMetrics;
					newDataSet.add(candidate);
					n++;
				}
			}
		}
		return new MultiLabelInstances(newDataSet, getXmlLabels());
	}

	private Integer numberNNSharingLabel(Instances nearNeigh) throws Exception {
		ArrayList<ArrayList<Attribute>> neighbors = new ArrayList<ArrayList<Attribute>>(); 
		for (Instance neighbor : nearNeigh) {
			ArrayList<Attribute> labels = new ArrayList<Attribute>();
			for (Integer index : getData().getLabelIndices()){
				Attribute label = neighbor.attribute(index);
				if (neighbor.value(label) == 1.0) {
					labels.add(label);
				}
			}
			neighbors.add(labels);
		}
		Integer count = 0;
		for (int i = 0; i < neighbors.size(); i++) {
			ArrayList<Attribute> labels1 = neighbors.get(i);
			for (int j = i; j < neighbors.size(); j++) {
				ArrayList<Attribute> labels2 = neighbors.get(j);
				List<Attribute> common = new ArrayList<Attribute>(labels1);
				common.retainAll(labels2);
				if (common.size() > 0) {
					count++;
				}
			}
		}
		return count;
	}

	private ArrayList<Instance> getUnvisitedInstances(ArrayList<Node> tree) {
		int i = 0;
		do {
			i++;
		} while (i < tree.size() && tree.get(i).wasVisited());
		if (i == tree.size()) {
			return new ArrayList<Instance>();
		}
		tree.get(i).setVisited(true);
		return tree.get(i).getInstances();
	}

	private ArrayList<Node> contructSortedTree() {
    	ArrayList<Node> tree = new ArrayList<Node>();
    	Map<Attribute, Double> sortedIRLbl = sortByValue(getMetrics().getIRLbl());
    	for (Attribute label : sortedIRLbl.keySet()) {
    		Node node = new Node();
    		node.setVisited(false);
    		node.setInstances(getInstancesFromLabel(label));
    		tree.add(node);
    	}
		return tree;
	}
    
	private ArrayList<Instance> getInstancesFromLabel(Attribute label) {
		ArrayList<Instance> instances = new ArrayList<Instance>(); 
		for (Instance instance : getData().getDataSet()) {
			if (instance.value(label) == 1.0) {
				instances.add(instance);
			}
		}
		return instances;
	}

    private <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            public int compare(Map.Entry<K, V> obj1, Map.Entry<K, V> obj2) {
                return (obj2.getValue()).compareTo(obj1.getValue());
            }
        });

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
}
