package mulan.resampling;

import java.util.ArrayList;

public class Bag implements Comparable<Bag>{
	private ArrayList<Integer> instances;
	private BagType type; 
	
	public Bag(ArrayList<Integer> instances, BagType type) {
		this.instances = instances;
		this.type = type;
	}
	
	public int compareTo(Bag compareBag) {
		int compareSize = ((Bag) compareBag).getInstances().size();
		if (getType().equals(BagType.Minority)) {
			return compareSize - this.getInstances().size();	
		} else {
			return this.getInstances().size() - compareSize;
		}
	}
	
	public boolean addInstance(Integer instance) {
		return instances.add(instance);
	}
	
	public void removeInstance(Integer instance) {
		instances.remove(instance);
	}
	
	public ArrayList<Integer> getInstances() {
		return instances;
	}

	public BagType getType() {
		return type;
	}

	public void setType(BagType type) {
		this.type = type;
	}
}