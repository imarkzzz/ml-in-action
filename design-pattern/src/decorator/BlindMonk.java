package decorator;

/**
 * Created by mark on 2018/10/26.
 */
public class BlindMonk implements Hero {
    private String name;

    public BlindMonk(String name) {
        this.name = name;
    }

    @Override
    public void learnSkills() {
        System.out.println(name + "学习了以上技能!");
    }
}
