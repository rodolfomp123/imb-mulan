package mulan.resampling;

public enum HybridMethod {
    MLROS(0),
    MLeNN(1),
    MLSMOTE(2);

    public int type;
    
    HybridMethod(int type) {
    	this.type = type;
    }
}
