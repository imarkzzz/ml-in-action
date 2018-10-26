package decorator;

/**
 * Created by mark on 2018/10/26.
 */
public class Skills implements Hero {

    private Hero hero;

    public Skills(Hero hero) {
        this.hero = hero;
    }

    @Override
    public void learnSkills() {
        if (hero != null) {
            hero.learnSkills();
        }
    }
}
