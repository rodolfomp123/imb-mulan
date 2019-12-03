package mulan.resampling.examples;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.resampling.HybridMethod;
import mulan.resampling.REMEDIALHwR;
import weka.core.Utils;

public class REMEDIALHwRExecuter {
	
    public static void main(String[] args) {
        try {
        	// e.g. -dir C:/Users/rodolfo/Desktop/workspace/mulan/data/
            String directory = Utils.getOption("dir", args);
        	// e.g. -arff emotions.arff
            String arffFilename = Utils.getOption("arff", args);
            // e.g. -xml emotions.xml
            String xmlFilename = Utils.getOption("xml", args);
            
            System.out.println("Loading the dataset...");
    		MultiLabelInstances originalDataset = new MultiLabelInstances(directory + arffFilename, directory + xmlFilename);
    		System.out.println("Resampling the dataset...");
    		REMEDIALHwR remedialhwr = new REMEDIALHwR(originalDataset, directory + xmlFilename, HybridMethod.MLSMOTE);
    		MultiLabelInstances resampledDataset = remedialhwr.resample();
    		System.out.println("New number of instances: " + resampledDataset.getNumInstances());
    		System.out.println("Writing result dataset...");
    		BufferedWriter writer = new BufferedWriter(new FileWriter(directory + "REMEDIALHwR-" + arffFilename));
    		writer.write(resampledDataset.getDataSet().toString());
    		writer.flush();
    		writer.close();
        } catch (InvalidDataFormatException ex) {
            Logger.getLogger(LPROSExecuter.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(LPROSExecuter.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
