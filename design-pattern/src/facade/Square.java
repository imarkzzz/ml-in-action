package facade;

/**
 * Created by mark on 2018/10/26.
 */
public class Square implements Shape {
    @Override
    public void draw() {
        System.out.println("Square::draw()");
    }
}
