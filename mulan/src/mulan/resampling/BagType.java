package mulan.resampling;

public enum BagType {
    Minority(0),
    Majority(1);

    public int type;
    
    BagType(int type) {
    	this.type = type;
    }
}