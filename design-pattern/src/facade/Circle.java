package facade;

/**
 * Created by mark on 2018/10/26.
 */
public class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Circle::draw()");
    }
}
