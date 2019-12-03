package mulan.resampling;

import java.util.ArrayList;

import weka.core.Instance;

public class Node {
    private Boolean visited;
    private ArrayList<Instance> instances;

	public Boolean wasVisited() {
		return visited;
	}

	public void setVisited(Boolean visited) {
		this.visited = visited;
	}
	
	public ArrayList<Instance> getInstances() {
		return instances;
	}

	public void setInstances(ArrayList<Instance> instances) {
		this.instances = instances;
	}
}
