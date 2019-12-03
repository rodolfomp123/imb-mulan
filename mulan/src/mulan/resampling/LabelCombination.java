package mulan.resampling;

public enum LabelCombination {
    Union(1),
    Intersection(2),
	Ranking(3);

    public int type;
    
    LabelCombination(int type) {
    	this.type = type;
    }
}