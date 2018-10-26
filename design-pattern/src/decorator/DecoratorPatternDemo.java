package decorator;

/**
 * Created by mark on 2018/10/26.
 */
public class DecoratorPatternDemo {
    public static void main(String[] args) {
        Hero blindMonk = new BlindMonk("Mark");
        blindMonk = new Skills(blindMonk);
        blindMonk = new Skill_R(blindMonk, "猛虎摆尾");
        blindMonk = new Skill_E(blindMonk, "天雷破");
        blindMonk = new Skill_W(blindMonk, "金钟罩");
        blindMonk = new Skill_Q(blindMonk, "天音破");
        blindMonk.learnSkills();
    }
}
