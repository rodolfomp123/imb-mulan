package mulan.evaluation;

import java.util.ArrayList;
import java.util.HashMap;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

public class HierarchicalEvaluation {

	private MultiLabelInstances train_dataset;
	private MultiLabelInstances test_dataset;
	private MultiLabelLearner learner;
	
	public HierarchicalEvaluation(
			 MultiLabelInstances train_dataset, 
			 MultiLabelInstances test_dataset, 
			 MultiLabelLearner learner) {
		this.train_dataset = train_dataset;
		this.test_dataset = test_dataset;
		this.learner = learner;
	}

	public Double[] evaluate() throws Exception {		
		learner.build(train_dataset);
        Instances testData = test_dataset.getDataSet();
        int numInstances = testData.numInstances();
        ArrayList<String> predicted = new ArrayList<String>();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            MultiLabelOutput output = learner.makePrediction(instance);
            double[] labelsRanking = output.getConfidences();
            String predictedLabels = ""; 
            for (int i = 0; i < labelsRanking.length; i++) {
				if (labelsRanking[i] >= 0.5) {
					String label = instance.attribute(i).name();
					if (predictedLabels != "") {
						predictedLabels += "@" + label;
					} else {
						predictedLabels += label;
					}
				}
			} 
            if (predictedLabels != "") {
            	predicted.add(predictedLabels);
            } else {
            	int indexFirst = output.getRanking()[0];
            	predicted.add(instance.attribute(indexFirst).name());
            }
        }
        Double[] measures = hierarquicalFscore(predicted);
        return measures;
    }
	
	private Double[] calcRatios(String[] pred_labels, String[] true_labels){
	    Double hits = 0.0;
	    for (int index = 0; index < pred_labels.length; index++) {
	    	String pred_label = pred_labels[index];
	        if (index < true_labels.length && true_labels[index].equals(pred_label)){
	            hits += 1.0;
	        } else {
	            break;
	        }
	    }
	    //recall,precision
	    Double[] ratios = new Double[2];
	    ratios[0] = hits/true_labels.length;
	    ratios[1] = hits/pred_labels.length;
	    return ratios;
	}   		
	    		
	public HashMap<String, ArrayList<Double>> createPathsDict(String[] label_paths){
		HashMap<String, ArrayList<Double>> pathsDict = new HashMap<String, ArrayList<Double>>();
	    for (String path : label_paths) {
	        pathsDict.put(path, new ArrayList<Double>());
	    }
	    return pathsDict;
	}
	
	public Double calcMean(ArrayList<Double> vector){
	    Double sumOfNums = 0.0D;
	    for (Double num : vector){
	        sumOfNums += num;
	    }
	    if (vector.size() == 0){
	        return null;
	    }
	    return sumOfNums / vector.size();
	}

	public HashMap<String, Double> countSamplesPerPath(ArrayList<String> dataset){
		HashMap<String, Double> count = new HashMap<String, Double>();
	    for (String label_paths : dataset) {
	        String[] paths = label_paths.split("@");
	        for (String path : paths){
	            if (count.containsKey(path)){
	                count.put(path, count.get(path) + 1);
	            } else {
	            	count.put(path, 1D);
	            }
	        }
	    }
	    return count;
	}
	
	public Double[] hierarquicalFscore(ArrayList<String> y_pred){
		
		ArrayList<String> y_test = retrieveTrueLabels();
		String[] label_paths = test_dataset.getLabelNames();
		
		HashMap<String, ArrayList<Double>> recallPathsDict = createPathsDict(label_paths);
		HashMap<String, ArrayList<Double>> precisionPathsDict = createPathsDict(label_paths);
	    for (int index = 0; index < y_pred.size(); index++) {
	    	String pred = y_pred.get(index);
	    	String true_l = y_test.get(index);
	        String[] pred_paths = pred.split("@");
	        String[] true_paths = true_l.split("@");
	        for (String pred_path : pred_paths) {
	            Double best_recall = -1.0;
	            Double best_precision = -1.0;
	            String best_path = "";
	            for (String true_path : true_paths) {
	            	String[] pred_labels = pred_path.split("/");
	            	String[] true_labels = true_path.split("/");
	                Double[] ratios = calcRatios(pred_labels, true_labels);
	                Double recall = ratios[0];
	                Double precision = ratios[1];
	                if (precision > best_precision) {
	                    best_precision = precision;
	                    best_path = true_path;
	                }
	                if (recall > best_recall) {
	                    best_recall = recall;
	                }
	            }
	            ArrayList<Double> newRecallList = recallPathsDict.get(best_path);
	            recallPathsDict.remove(best_path);
	            newRecallList.add(best_recall);
	            recallPathsDict.put(best_path, newRecallList);
	            ArrayList<Double> newPrecisionList = precisionPathsDict.get(best_path);
	            precisionPathsDict.remove(best_path);
	            newPrecisionList.add(best_precision);
	            precisionPathsDict.put(best_path, newPrecisionList);
	    	}
	    }
	    HashMap<String, Double> recallPerClass = new HashMap<String, Double>();
	    HashMap<String, Double> precisionPerClass = new HashMap<String, Double>();
	    for (String path : label_paths) {
	        recallPerClass.put(path, calcMean(recallPathsDict.get(path))); 
	        precisionPerClass.put(path, calcMean(precisionPathsDict.get(path)));
	    }
	    Double sum_precision = 0D;
	    Double total_p = 0D;
	    Double sum_recall = 0D;
	    Double total_r = 0D;
	    for (String path : label_paths) {
	        if (recallPerClass.get(path) != null){
	            sum_recall += recallPerClass.get(path);
	        }
	        total_r += 1.0;
	        if (precisionPerClass.get(path) != null) {
	            sum_precision += precisionPerClass.get(path);
	        }
	        total_p += 1.0;
	    }
	    Double recall = sum_recall/total_r;
	    Double precision = sum_precision/total_p;
	    Double fscore;
	    if ((recall + precision) == 0D) {
	        fscore = 0D;
	    } else {
	        fscore = (2*recall*precision)/(recall+precision);
	    }
	    Double[] measures = new Double[3];
	    measures[0] = fscore;
	    measures[1] = recall;
	    measures[2] = precision;
	    return measures;
	}

	private ArrayList<String> retrieveTrueLabels() {
		Instances testData = test_dataset.getDataSet();
		int[] labelIndices = test_dataset.getLabelIndices();
		ArrayList<String> trueLabels = new ArrayList<String>();
		int numInstances = testData.numInstances();
		for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
			String labelSet = "";
			Instance instance = testData.instance(instanceIndex);
			for (int index : labelIndices) {
				Double attr = instance.value(index);
				if (attr == 1.0) {
					String label = instance.attribute(index).name();
					if (labelSet != "") {
						labelSet += "@" + label;
					} else {
						labelSet += label;
					}
				}
			}
			trueLabels.add(labelSet);
		}
		return trueLabels;
	}
}
